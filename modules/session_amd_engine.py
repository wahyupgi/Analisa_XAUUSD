# -*- coding: utf-8 -*-
"""
modules/session_amd_engine.py
Trading Sessions + Power of Three (PO3) / AMD (Accumulation - Manipulation - Distribution) Engine.

PRINSIP:
- 100% Python deterministik (OHLC candle CLOSED saja, tanpa look-ahead, tanpa AI).
- Fase AMD ditentukan oleh PERGERAKAN HARGA AKTUAL, BUKAN oleh jam sesi:
    * ACCUMULATION     : kompresi/range sempit (lebar <= k x ATR, efisiensi pergerakan rendah) yang belum ditembus.
    * MANIPULATION     : sweep (false break) di luar range / level likuiditas lalu CLOSE kembali ke dalam (reclaim).
    * DISTRIBUTION     : setelah manipulation, displacement + progres berlawanan arah sweep.
    * DIRECT_EXPANSION : breakout yang bertahan di luar tanpa reclaim (BUKAN manipulation; jangan dilawan).
    * NO_CLEAR_AMD     : tidak ada pola valid (tidak dipaksa diberi label palsu).
- Sesi (Asia/London/NY/Overlap) hanya KONTEKS waktu & sumber level likuiditas (DST-aware). Sesi TIDAK
  dipakai untuk menebak fase; fase yang muncul di sesi non-klasik hanya diberi catatan informasional.
- Modul ini tidak mengubah skor zona, risk, lot, maupun order. Satu-satunya efek eksekusi: menahan
  EARLY_PENDING (limit buta sebelum zona tersentuh) yang berisiko -> tetap bisa lewat ADAPTIVE_ENTRY
  saat zona tersentuh + trigger. Data tidak tersedia / error => TIDAK memblokir (indikator opsional).
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.config import get_amd_config
from modules.execution_state import REASON_AMD_COUNTER_PHASE, REASON_AMD_LIQUIDITY_EXPOSED
from modules.utils import us_dst

PHASE_ACCUMULATION = "ACCUMULATION"
PHASE_MANIPULATION = "MANIPULATION"
PHASE_DISTRIBUTION = "DISTRIBUTION"
PHASE_DIRECT_EXPANSION = "DIRECT_EXPANSION"
PHASE_NO_CLEAR = "NO_CLEAR_AMD"
PHASE_UNAVAILABLE = "UNAVAILABLE"

REL_ALIGNED_POST_SWEEP = "ALIGNED_POST_SWEEP"
REL_ALIGNED_EXPANSION = "ALIGNED_EXPANSION"
REL_COUNTER_PHASE = "COUNTER_PHASE"
REL_LIQUIDITY_EXPOSED = "LIQUIDITY_EXPOSED"
REL_NEUTRAL = "NEUTRAL"

_SRC_MIN = 5            # frame sumber selalu M5 (closed)
_BOX_WINDOWS = (32, 20, 12)   # panjang kandidat kotak akumulasi (bar M15): 8 jam, 5 jam, 3 jam
_CLASSIC = {"ACCUMULATION": "ASIA", "MANIPULATION": "LONDON", "DISTRIBUTION": "NEW_YORK"}

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"key": None, "val": None}


# =============================================================================
# WAKTU & SESI (DST-aware, semua dalam UTC naive)
# =============================================================================
def _last_sunday(year: int, month: int) -> datetime:
    d = datetime(year, month, 31)   # Maret & Oktober punya 31 hari
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d


def uk_dst(dt: datetime) -> bool:
    """UK BST: Minggu terakhir Maret 01:00 UTC s/d Minggu terakhir Oktober 01:00 UTC."""
    return _last_sunday(dt.year, 3) + timedelta(hours=1) <= dt < _last_sunday(dt.year, 10) + timedelta(hours=1)


def session_bounds_utc(dt: datetime) -> Dict[str, float]:
    """Batas sesi (jam UTC) pada tanggal dt: London open, NY open, London close, NY close."""
    ref = datetime(dt.year, dt.month, dt.day, 12)
    uk = 1 if uk_dst(ref) else 0
    us = 1 if us_dst(ref) else 0
    return {"london_open": 8.0 - uk, "ny_open": 13.0 - us, "london_close": 17.0 - uk, "ny_close": 22.0 - us}


def session_at(ts: datetime) -> str:
    """Sesi aktif pada waktu ts (UTC naive): ASIA / LONDON / LONDON_NY_OVERLAP / NEW_YORK / OFF_HOURS."""
    b = session_bounds_utc(ts)
    h = ts.hour + ts.minute / 60.0
    if h < b["london_open"]:
        return "ASIA"
    if h < b["ny_open"]:
        return "LONDON"
    if h < b["london_close"]:
        return "LONDON_NY_OVERLAP"
    if h < b["ny_close"]:
        return "NEW_YORK"
    return "OFF_HOURS"


def _kill_zone(ts: datetime) -> Optional[str]:
    """Kill zone (jam New York) sama dengan konvensi analisa_sesi: London 02-05, NY 07-10, London Close 10-12."""
    ny = ts + timedelta(hours=-4 if us_dst(ts) else -5)
    h = ny.hour + ny.minute / 60.0
    if h >= 20 or h < 0:
        return "Asia Kill Zone"
    if 2 <= h < 5:
        return "London Open Kill Zone"
    if 7 <= h < 10:
        return "New York Open Kill Zone"
    if 10 <= h < 12:
        return "London Close Kill Zone"
    return None


def _index_utc(df: pd.DataFrame, server_offset_hours: float = 0.0) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    hint = str((getattr(df, "attrs", {}) or {}).get("time_tz", "UTC")).upper()
    if hint == "WIB":
        return idx - pd.Timedelta(hours=7)
    if server_offset_hours:
        return idx - pd.Timedelta(hours=float(server_offset_hours))
    return idx


# =============================================================================
# DATA: M5/M15 closed -> M15 closed + ATR
# =============================================================================
def _build_frame(
    df_m5: Optional[pd.DataFrame],
    as_of: datetime,
    tf_min: int,
    df_m15: Optional[pd.DataFrame] = None,
    server_offset_hours: float = 0.0,
) -> Optional[pd.DataFrame]:
    cols = ("Open", "High", "Low", "Close")
    parts: List[pd.DataFrame] = []
    if int(tf_min) == 15 and df_m15 is not None and len(df_m15) > 0 and all(c in df_m15.columns for c in cols):
        x15 = pd.DataFrame(
            {c: pd.to_numeric(df_m15[c].values, errors="coerce") for c in cols},
            index=_index_utc(df_m15, server_offset_hours=server_offset_hours),
        )
        x15 = x15[~x15.index.duplicated(keep="last")].sort_index().dropna()
        x15 = x15[x15.index + pd.Timedelta(minutes=15) <= pd.Timestamp(as_of)]
        if len(x15) > 0:
            parts.append(x15)

    if df_m5 is not None and len(df_m5) > 0 and all(c in df_m5.columns for c in cols):
        x = pd.DataFrame(
            {c: pd.to_numeric(df_m5[c].values, errors="coerce") for c in cols},
            index=_index_utc(df_m5, server_offset_hours=server_offset_hours),
        )
        x = x[~x.index.duplicated(keep="last")].sort_index().dropna()
        x = x[x.index + pd.Timedelta(minutes=_SRC_MIN) <= pd.Timestamp(as_of)]   # hanya M5 yang sudah CLOSED
        if len(x) > 0:
            agg = x.resample(f"{int(tf_min)}min", label="left", closed="left").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            ).dropna()
            agg = agg[agg.index + pd.Timedelta(minutes=int(tf_min)) <= pd.Timestamp(as_of)]
            if len(agg) > 0:
                parts.append(agg)

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    merged = pd.concat(parts, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged if len(merged) else None


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    if len(c) < period + 1:
        return 0.0
    pc = c[:-1]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    v = float(np.mean(tr[-period:]))
    return v if np.isfinite(v) and v > 0 else 0.0


# =============================================================================
# EVALUATOR SWEEP / RECLAIM / DISTRIBUTION / EXPANSION TERHADAP SATU REFERENSI
# =============================================================================
def _evaluate_reference(o, h, l, c, start: int, ref_hi: Optional[float], ref_lo: Optional[float],
                        atr: float, p: Dict[str, Any]) -> Dict[str, Any]:
    """Cari penembusan pertama referensi (box/level) setelah bar `start`, lalu klasifikasikan dari harga aktual."""
    n = len(c)
    pen = float(p["AMD_SWEEP_MIN_PEN_ATR"]) * atr
    b, side = -1, None
    for i in range(start + 1, n):
        up = ref_hi is not None and h[i] > ref_hi + pen
        dn = ref_lo is not None and l[i] < ref_lo - pen
        if up and dn:
            side = "HIGH" if (h[i] - ref_hi) >= (ref_lo - l[i]) else "LOW"
        elif up:
            side = "HIGH"
        elif dn:
            side = "LOW"
        else:
            continue
        b = i
        break
    if side is None:
        return {"state": "NO_BREACH"}
    if (n - 1 - b) > int(p["AMD_MAX_EVENT_AGE_BARS"]):
        return {"state": "STALE"}

    high = side == "HIGH"
    ref = ref_hi if high else ref_lo
    last_k = min(n - 1, b + int(p["AMD_RECLAIM_MAX_BARS"]))
    reclaim = -1
    for k in range(b, last_k + 1):
        if (c[k] <= ref) if high else (c[k] >= ref):
            reclaim = k
            break

    if reclaim >= 0:
        ext = float(np.max(h[b:reclaim + 1])) if high else float(np.min(l[b:reclaim + 1]))
        if abs(ext - ref) > float(p["AMD_SWEEP_MAX_EXC_ATR"]) * atr:
            return {"state": "AMBIGUOUS", "reason": "sweep terlalu dalam (bukan false break)"}
        if (c[-1] > ext) if high else (c[-1] < ext):
            # harga menutup melewati ekstrem sweep => sweep gagal, berubah menjadi ekspansi searah penembusan
            return {"state": "EVENT", "phase": PHASE_DIRECT_EXPANSION, "bias": "BULLISH" if high else "BEARISH",
                    "side": side, "ref": float(ref), "extreme": ext, "b": b, "reclaim": reclaim,
                    "reason": f"sweep gagal: close {c[-1]:.2f} melewati ekstrem sweep {ext:.2f}"}
        seg_c = c[reclaim:]
        progress = float(ref - np.min(seg_c)) if high else float(np.max(seg_c) - ref)
        body_thr = float(p["AMD_DISP_BODY_ATR"]) * atr
        bodies = (o[reclaim:] - c[reclaim:]) if high else (c[reclaim:] - o[reclaim:])
        disp = bool(np.max(bodies) >= body_thr)
        still_ok = (c[-1] < ref) if high else (c[-1] > ref)
        bias = "BEARISH" if high else "BULLISH"
        if disp and still_ok and progress >= float(p["AMD_DIST_MIN_ATR"]) * atr:
            return {"state": "EVENT", "phase": PHASE_DISTRIBUTION, "bias": bias, "side": side, "ref": float(ref),
                    "extreme": ext, "b": b, "reclaim": reclaim, "progress": round(progress, 2),
                    "reason": (f"sweep {'high' if high else 'low'} {ref:.2f} (ekstrem {ext:.2f}) -> reclaim -> "
                               f"displacement {bias.lower()} {progress:.2f} USD")}
        return {"state": "EVENT", "phase": PHASE_MANIPULATION, "bias": bias, "side": side, "ref": float(ref),
                "extreme": ext, "b": b, "reclaim": reclaim, "progress": round(max(progress, 0.0), 2),
                "reason": f"sweep {'high' if high else 'low'} {ref:.2f} (ekstrem {ext:.2f}) lalu close kembali ke dalam"}

    # Tidak ada reclaim dalam jendela
    beyond = (c[-1] - ref) if high else (ref - c[-1])
    if beyond >= float(p["AMD_EXPANSION_CONFIRM_ATR"]) * atr:
        return {"state": "EVENT", "phase": PHASE_DIRECT_EXPANSION, "bias": "BULLISH" if high else "BEARISH",
                "side": side, "ref": float(ref), "extreme": float(np.max(h[b:]) if high else np.min(l[b:])),
                "b": b, "reclaim": -1,
                "reason": f"breakout {'atas' if high else 'bawah'} {ref:.2f} bertahan di luar tanpa reclaim"}
    return {"state": "AMBIGUOUS", "reason": "penembusan belum terselesaikan (belum reclaim / belum bertahan)"}


# =============================================================================
# LEVEL LIKUIDITAS SESI
# =============================================================================
def _build_levels(t: pd.DatetimeIndex, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                  as_of: datetime) -> List[Dict[str, Any]]:
    day0 = datetime(as_of.year, as_of.month, as_of.day)
    prev0 = day0 - timedelta(days=1)
    b = session_bounds_utc(day0)
    defs = []
    # Previous Day High/Low hanya bila data menutup hari sebelumnya secara memadai
    if len(t) and t[0] <= pd.Timestamp(prev0 + timedelta(hours=2)):
        defs.append(("PDH", "PDL", prev0, day0, day0, True))
    asia_end = day0 + timedelta(hours=b["london_open"])
    lon_end = day0 + timedelta(hours=b["ny_open"])
    defs.append(("Asia High", "Asia Low", day0, asia_end, asia_end, as_of >= asia_end))
    defs.append(("London High", "London Low", asia_end, lon_end, lon_end, as_of >= lon_end))

    out: List[Dict[str, Any]] = []
    for nm_h, nm_l, a, z, est_end, complete in defs:
        m = (t >= pd.Timestamp(a)) & (t < pd.Timestamp(z))
        if int(m.sum()) < 4:
            continue
        hi, lo = float(np.max(h[m])), float(np.min(l[m]))
        for nm, price, kind in ((nm_h, hi, "H"), (nm_l, lo, "L")):
            lv = {"name": nm, "price": round(price, 2), "kind": kind, "est_end": est_end,
                  "complete": bool(complete), "status": "FORMING", "start": -1}
            if complete:
                post = t >= pd.Timestamp(est_end)
                lv["start"] = int(np.searchsorted(t.values, np.datetime64(est_end), side="left")) - 1
                if not post.any():
                    lv["status"] = "UNTOUCHED"
                else:
                    ph, pl, last_c = h[post], l[post], c[-1]
                    if kind == "H":
                        lv["status"] = ("BROKEN" if last_c > price else "SWEPT") if np.max(ph) > price else "UNTOUCHED"
                    else:
                        lv["status"] = ("BROKEN" if last_c < price else "SWEPT") if np.min(pl) < price else "UNTOUCHED"
            out.append(lv)
    return out


# =============================================================================
# ANALISIS UTAMA
# =============================================================================
def _unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "phase": PHASE_UNAVAILABLE, "bias": "NEUTRAL", "reason": reason,
            "levels": [], "session": None, "atr": 0.0}


def analyze_session_amd(
    df_m5: Optional[pd.DataFrame],
    as_of: Optional[datetime] = None,
    cfg: Optional[Dict[str, Any]] = None,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    server_offset_hours: float = 0.0,
) -> Dict[str, Any]:
    """Analisis sesi + fase PO3/AMD dari candle M5/M15 CLOSED. Tidak pernah melempar exception."""
    try:
        p = dict(get_amd_config())
        if cfg:
            p.update({k: v for k, v in cfg.items() if str(k).startswith("AMD_")})
        if not bool(p.get("AMD_ENGINE_ENABLED", True)):
            return _unavailable("AMD engine dimatikan (AMD_ENGINE_ENABLED=false)")
        df_m15 = (frames or {}).get("M15") if isinstance(frames, dict) else None
        ref_df = df_m5 if (df_m5 is not None and len(df_m5) > 0) else df_m15
        if ref_df is None or len(ref_df) == 0:
            return _unavailable("data M5 tidak tersedia")
        ref_step = _SRC_MIN if (df_m5 is not None and len(df_m5) > 0) else 15
        idx_utc = _index_utc(ref_df, server_offset_hours=server_offset_hours)
        as_of_u = as_of if as_of is not None else (idx_utc[-1].floor("s").to_pydatetime() + timedelta(minutes=ref_step))
        sig = tuple(round(float(pd.to_numeric(ref_df[k], errors="coerce").sum()), 4) for k in ("High", "Low", "Close"))
        n15 = len(df_m15) if (df_m15 is not None) else 0
        key = (len(ref_df), n15, round(float(server_offset_hours), 2), idx_utc[0], idx_utc[-1], as_of_u, sig, tuple(sorted((k, str(v)) for k, v in p.items())))
        with _CACHE_LOCK:
            if _CACHE["key"] == key:
                return _CACHE["val"]
        res = _analyze(df_m5, as_of_u, p, df_m15=df_m15, server_offset_hours=server_offset_hours)
        with _CACHE_LOCK:
            _CACHE["key"], _CACHE["val"] = key, res
        return res
    except Exception as e:  # opsional: tidak boleh mengganggu scanner/eksekusi
        return _unavailable(f"error AMD engine: {type(e).__name__}: {e}")


def _analyze(
    df_m5: Optional[pd.DataFrame],
    as_of: datetime,
    p: Dict[str, Any],
    df_m15: Optional[pd.DataFrame] = None,
    server_offset_hours: float = 0.0,
) -> Dict[str, Any]:
    tf_min = int(p["AMD_ANALYSIS_TF_MIN"])
    fr = _build_frame(df_m5, as_of, tf_min, df_m15=df_m15, server_offset_hours=server_offset_hours)
    if fr is None or len(fr) < int(p["AMD_MIN_BARS"]):
        return _unavailable(f"data M{tf_min} tidak cukup (min {int(p['AMD_MIN_BARS'])} bar)")
    o, h, l, c = (fr[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    t = fr.index
    n = len(c)
    atr = _atr(h, l, c)
    if atr <= 0:
        return _unavailable("ATR tidak valid")

    bounds = session_bounds_utc(as_of)
    sess = {
        "active": session_at(as_of), "kill_zone": _kill_zone(as_of), "as_of_utc": as_of.strftime("%Y-%m-%d %H:%M"),
        "london_open_utc": bounds["london_open"], "ny_open_utc": bounds["ny_open"],
    }
    levels = _build_levels(t, h, l, c, as_of)

    events: List[Dict[str, Any]] = []     # kandidat event (box & level)
    acc: List[Dict[str, Any]] = []        # kandidat akumulasi aktif

    # --- Kandidat kotak akumulasi: kompresi + efisiensi rendah; evaluasi bar SETELAH kotak (tanpa look-ahead)
    max_age = int(p["AMD_MAX_EVENT_AGE_BARS"])
    for W in _BOX_WINDOWS:
        for e in range(n - 1, max(W - 2, n - 2 - max_age), -1):
            s = e - W + 1
            hi, lo = float(np.max(h[s:e + 1])), float(np.min(l[s:e + 1]))
            width = hi - lo
            if width > float(p["AMD_ACC_RANGE_ATR"]) * atr or width < float(p["AMD_ACC_MIN_RANGE_ATR"]) * atr:
                continue
            # Kemurnian kotak: ekor (3 bar terakhir) tidak boleh memperluas inti kotak. Jika memperluas, ekor itu adalah
            # penembusan/sweep yang HARUS dievaluasi terhadap inti, bukan disembunyikan di dalam kotak.
            core_hi, core_lo = float(np.max(h[s:e - 2])), float(np.min(l[s:e - 2]))
            pen_box = float(p["AMD_SWEEP_MIN_PEN_ATR"]) * atr
            if float(np.max(h[e - 2:e + 1])) > core_hi + pen_box or float(np.min(l[e - 2:e + 1])) < core_lo - pen_box:
                continue
            path = float(np.sum(np.abs(np.diff(c[s:e + 1]))))
            if path <= 0 or abs(c[e] - c[s]) / path > float(p["AMD_ACC_MAX_EFFICIENCY"]):
                continue
            ev = _evaluate_reference(o, h, l, c, e, hi, lo, atr, p)
            box = {"hi": round(hi, 2), "lo": round(lo, 2), "start_time": t[s], "end_time": t[e], "bars": W}
            if ev["state"] == "NO_BREACH":
                acc.append({"e": e, "W": W, "box": box})
            elif ev["state"] == "EVENT":
                ev.update(source="BOX", label=f"Range {lo:.2f}-{hi:.2f}", box=box, rank=(ev["b"], 1, W))
                events.append(ev)

    # --- Kandidat sweep level likuiditas sesi (hanya level yang sudah final)
    for lv in levels:
        if not lv["complete"] or lv["start"] < 0 or lv["start"] >= n - 1:
            continue
        ev = _evaluate_reference(o, h, l, c, lv["start"],
                                 lv["price"] if lv["kind"] == "H" else None,
                                 lv["price"] if lv["kind"] == "L" else None, atr, p)
        if ev["state"] == "EVENT":
            ev.update(source="LEVEL", label=lv["name"], box=None, rank=(ev["b"], 0, 0))
            events.append(ev)

    out: Dict[str, Any] = {"available": True, "atr": round(atr, 2), "session": sess, "levels": [
        {"name": x["name"], "price": x["price"], "status": x["status"], "kind": x["kind"]} for x in levels],
        "analysis_tf": f"M{tf_min}", "bars": n, "box": None, "event": None}

    if events:
        ev = max(events, key=lambda x: x["rank"])
        if ev["phase"] == PHASE_DIRECT_EXPANSION:
            # Ekspansi terbaru yang searah dengan manipulation sebelumnya = kelanjutan DISTRIBUTION (mis. sweep high ->
            # reclaim -> turun menembus Asia Low). Ekspansi berlawanan bias manipulation = sweep gagal -> tetap ekspansi.
            prior = [x for x in events if x["phase"] in (PHASE_MANIPULATION, PHASE_DISTRIBUTION)
                     and x["bias"] == ev["bias"] and x["b"] <= ev["b"]]
            if prior:
                m = max(prior, key=lambda x: x["rank"])
                ev = dict(m, phase=PHASE_DISTRIBUTION,
                          reason=f"{m['reason']} -> diteruskan ekspansi menembus {ev['label']} {ev['ref']:.2f}")
        ev_time = t[ev["b"]].to_pydatetime()
        ev_sess = session_at(ev_time)
        classic = _CLASSIC.get(ev["phase"])
        note = ""
        if ev["phase"] in _CLASSIC:
            note = ("sesuai pola klasik" if ev_sess == classic or (classic == "NEW_YORK" and ev_sess == "LONDON_NY_OVERLAP")
                    else f"NON-KLASIK: {ev['phase']} terjadi di sesi {ev_sess} (fase ditentukan oleh harga, bukan jam)")
        out.update(
            phase=ev["phase"], bias=ev["bias"], reference=ev["label"], source=ev["source"], side=ev["side"],
            ref_price=round(ev["ref"], 2), extreme=round(ev["extreme"], 2), reason=ev["reason"],
            box=ev.get("box"), event={"time_utc": ev_time.strftime("%Y-%m-%d %H:%M"), "session": ev_sess,
                                      "bars_ago": n - 1 - ev["b"], "note": note},
        )
        return out

    if acc:
        a = max(acc, key=lambda x: (x["e"], x["W"]))
        bx = a["box"]
        out.update(phase=PHASE_ACCUMULATION, bias="NEUTRAL", reference=f"Range {bx['lo']:.2f}-{bx['hi']:.2f}",
                   source="BOX", box=bx, reason=f"kompresi {bx['bars']} bar M{tf_min} belum ditembus "
                   f"(lebar {bx['hi'] - bx['lo']:.2f} USD = {(bx['hi'] - bx['lo']) / atr:.1f} ATR)")
        return out

    out.update(phase=PHASE_NO_CLEAR, bias="NEUTRAL", reason="tidak ada kompresi/sweep/ekspansi valid (tidak dipaksa label AMD)")
    return out


# =============================================================================
# RELASI ZONA <-> AMD (dipakai entry engine untuk EARLY_PENDING)
# =============================================================================
def evaluate_zone_amd_alignment(amd: Optional[Dict[str, Any]], zone_type: str, zone_low: float, zone_high: float,
                                cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Relasi zona dengan fase AMD aktual. `hold_early_pending` hanya menahan limit buta pra-sentuh."""
    neutral = {"relation": REL_NEUTRAL, "hold_early_pending": False, "reason_code": "", "reason": ""}
    try:
        if not isinstance(amd, dict) or not amd.get("available"):
            return neutral
        p = dict(get_amd_config())
        if cfg:
            p.update({k: v for k, v in cfg.items() if str(k).startswith("AMD_")})
        demand = str(zone_type).upper() == "DEMAND"
        want = "BULLISH" if demand else "BEARISH"
        phase, bias = amd.get("phase"), amd.get("bias")
        ref = amd.get("reference") or "-"

        if phase in (PHASE_MANIPULATION, PHASE_DISTRIBUTION) and bias == want:
            return {**neutral, "relation": REL_ALIGNED_POST_SWEEP,
                    "reason": f"zona searah {phase} pasca-sweep {ref}"}
        if phase == PHASE_DIRECT_EXPANSION and bias == want:
            return {**neutral, "relation": REL_ALIGNED_EXPANSION, "reason": f"zona searah ekspansi {ref}"}
        if phase in (PHASE_MANIPULATION, PHASE_DISTRIBUTION, PHASE_DIRECT_EXPANSION) and bias in ("BULLISH", "BEARISH"):
            return {"relation": REL_COUNTER_PHASE, "hold_early_pending": True, "reason_code": REASON_AMD_COUNTER_PHASE,
                    "reason": (f"Limit awal ditahan: fase {phase} {bias} ({ref}) melawan zona "
                               f"{'DEMAND' if demand else 'SUPPLY'} — tunggu sentuh zona + trigger M5")}

        atr = float(amd.get("atr") or 0.0)
        near = float(p["AMD_LIQUIDITY_NEAR_ATR"]) * atr
        lo, hi = float(zone_low), float(zone_high)
        if near > 0:
            cands = []
            box = amd.get("box")
            if phase == PHASE_ACCUMULATION and isinstance(box, dict):
                cands.append(("range accumulation", float(box["lo"]) if demand else float(box["hi"])))
            for lv in amd.get("levels") or []:
                if lv.get("status") == "UNTOUCHED" and lv.get("kind") == ("L" if demand else "H"):
                    cands.append((lv["name"], float(lv["price"])))
            for nm, px in cands:
                gap = (lo - px) if demand else (px - hi)    # likuiditas berada tepat di balik zona
                if 0 < gap <= near:
                    return {"relation": REL_LIQUIDITY_EXPOSED, "hold_early_pending": True,
                            "reason_code": REASON_AMD_LIQUIDITY_EXPOSED,
                            "reason": (f"Limit awal ditahan: likuiditas {nm} {px:.2f} belum disapu tepat di balik zona "
                                       f"(jarak {gap:.2f} USD) — tunggu sweep/reclaim atau sentuh zona + trigger M5")}
        return neutral
    except Exception:
        return neutral


# =============================================================================
# SCAN LIVE (MT5 read-only) & FORMAT RINGKAS
# =============================================================================
def scan_session_amd(df_m5: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Ambil M5/M15 closed dari MT5 (read-only) bila df tidak diberikan, lalu analisis dengan koreksi offset server broker."""
    srv_off = 0.0
    frames_live: Optional[Dict[str, pd.DataFrame]] = None
    if df_m5 is None:
        try:
            import MetaTrader5 as mt5
            from modules.config import SERVER_UTC_OFFSET_JAM, SYMBOL, TIMEFRAMES
            from modules.mt5_conn import MT5_IPC_LOCK, ambil_data, pastikan_mt5_aktif
            from modules.utils import deteksi_offset
            pastikan_mt5_aktif()
            df_m5 = ambil_data(TIMEFRAMES["M5"][0], 700)
            df_m15 = ambil_data(TIMEFRAMES["M15"][0], 300)
            frames_live = {"M5": df_m5, "M15": df_m15} if df_m15 is not None else None
            if SERVER_UTC_OFFSET_JAM is not None:
                srv_off = float(SERVER_UTC_OFFSET_JAM)
            else:
                with MT5_IPC_LOCK:
                    tk = mt5.symbol_info_tick(SYMBOL)
                srv_off = float(deteksi_offset(tk, df_m5) or 0.0)
        except Exception as e:
            return _unavailable(f"MT5 tidak tersedia: {type(e).__name__}")
    return analyze_session_amd(df_m5, frames=frames_live, server_offset_hours=srv_off)


def format_session_amd(amd: Dict[str, Any], detailed: bool = True) -> str:
    if not amd or not amd.get("available"):
        return f"SESI & AMD: UNAVAILABLE — {(amd or {}).get('reason', 'tidak ada data')}"
    s = amd.get("session") or {}
    ev = amd.get("event") or {}
    head = (f"SESI & AMD: sesi {s.get('active')} | {amd['phase']}"
            + (f" ({amd['bias']})" if amd.get("bias") in ("BULLISH", "BEARISH") else ""))
    if not detailed:
        return head + f" | {amd.get('reference', '-')}"
    lines = ["===== SESI & POWER OF THREE / AMD (berdasarkan pergerakan harga aktual) =====",
             f"Waktu: {s.get('as_of_utc')} UTC | Sesi aktif: {s.get('active')} | Kill Zone: {s.get('kill_zone') or '-'}",
             f"Fase AMD: {amd['phase']}" + (f" | Bias: {amd['bias']}" if amd.get("bias") in ("BULLISH", "BEARISH") else "")
             + f" | Referensi: {amd.get('reference', '-')}",
             f"Alasan: {amd.get('reason')}"]
    if ev:
        lines.append(f"Event: {ev.get('time_utc')} UTC ({ev.get('bars_ago')} bar {amd.get('analysis_tf')} lalu) di sesi "
                     f"{ev.get('session')}" + (f" — {ev['note']}" if ev.get("note") else ""))
    lv = amd.get("levels") or []
    if lv:
        lines.append("Level likuiditas: " + " | ".join(f"{x['name']} {x['price']:.2f} [{x['status']}]" for x in lv))
    return "\n".join(lines)


def context_lines() -> List[str]:
    """Satu baris ringkas untuk header scan/pantau (aman dari error)."""
    try:
        return [format_session_amd(scan_session_amd(), detailed=False)]
    except Exception:
        return []
