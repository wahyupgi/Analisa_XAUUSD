# -*- coding: utf-8 -*-
"""
modules/trade_validator.py
Validator Terpadu 7-Lapis untuk XAUUSD SMC Adaptive Auto Trading System:
1. Mandatory Hard Filters (MT5, Account, Real Auth, Market Open, Data Freshness, News, Spread, Duplicate)
2. Zone Quality Validation (Active, Score >= 70, Freshness/Mitigation, Structural Invalidation)
3. Directional Validation (H4/H1/M30 rules per jalur entry + evaluasi struktural SMC vs SNR)
4. Entry Trigger Validation (OR logic Adaptive, Confirmed, atau Early Pending + Anti-Chase)
5. Supporting Confluence (HTF S/D, FVG, Sweep, SNR — tanpa menjadikan indikator opsional sebagai blocker)
6. Risk Validation & Dynamic Lot Selection (Risk budget efektif, SL/TP, RR aktual >= 1.5, Lot 0.01/0.02/0.05)
7. Execution Validation (Broker stops/freeze, volume step, margin)

Memisahkan secara eksplisit: MARKET BIAS, TRADE PLAN, ENTRY STATUS, dan EXECUTION STATUS.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from modules import analisis_pasar, us_news_radar
from modules.adaptive_entry_engine import evaluate_entry_paths_for_zone, _ensure_zone_obj
from modules.config import SYMBOL, get_auto_trading_config
from modules.dynamic_lot_manager import (
    classify_zone_quality_label,
    resolve_dynamic_lot_for_trade,
    validate_strong_zone,
)
from modules.execution_state import (
    ExecutionStateManager,
    REASON_DIRECTION_NOT_ALIGNED,
    REASON_DUPLICATE_SETUP,
    REASON_HTF_CONFLICT,
    REASON_MARKET_CLOSED,
    REASON_MT5_DISCONNECTED,
    REASON_NEWS_BLACKOUT,
    REASON_OK,
    REASON_PRICE_LEFT_ENTRY_RANGE,
    REASON_SPREAD_TOO_HIGH,
    REASON_STALE_MARKET_DATA,
    REASON_STALE_NEWS_DATA,
    REASON_STOPPED_OUT_SETUP,
    REASON_WAIT_TRIGGER,
    REASON_WAIT_ZONE_TOUCH,
    REASON_ZONE_EXPIRED,
    REASON_ZONE_INVALIDATED,
    REASON_ZONE_LOW_SCORE,
    REASON_ZONE_MITIGATED,
    REASON_ZONE_TOO_FAR,
    STATE_APPROACHING,
    STATE_BLOCKED,
    STATE_ENTRY_READY,
    STATE_EXPIRED,
    STATE_INVALIDATED,
    STATE_ORDER_PLACED,
    STATE_POSITION_ACTIVE,
    STATE_RADAR,
    STATE_WAIT_TRIGGER,
    STATE_ZONE_TOUCHED,
    make_setup_id,
)
from modules.risk_manager import (
    compute_effective_risk_budget,
    evaluate_portfolio_risk_and_daily_limits,
    inspect_mt5_account_and_permissions,
    validate_sl_tp_and_rr,
)
from modules.trend_following import evaluate_snr_opposing_barrier
from modules.zone_scanner import (
    evaluate_smc_setup_live_invalidation,
    evaluate_smc_zone_closed_invalidation,
    is_far_reference_zone,
)


def evaluate_news_hard_filter(
    news_override: Optional[Dict[str, Any]] = None,
    snapshot_news: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Evaluasi filter berita High-Impact USD & kesegaran data berita (Section 6 & 11):
    - Jika high-impact news berada dalam blackout window -> BLOCK (NEWS_BLACKOUT)
    - Jika BLOCK_ON_STALE_NEWS_DATA=true dan berita STALE / UNAVAILABLE -> BLOCK (STALE_NEWS_DATA)
    Mendukung `news_override` deterministik untuk unit testing & simulasi.
    """
    conf = cfg or get_auto_trading_config()
    if not conf.get("NEWS_FILTER", True):
        return True, REASON_OK, "News filter disabled"

    block_on_stale = bool(conf.get("BLOCK_ON_STALE_NEWS_DATA", True))
    blackout_min = float(conf.get("BERITA_BLACKOUT_MENIT", 15.0))
    max_stale_sec = float(conf.get("NEWS_STALE_MAX_SECONDS", 21600.0))

    if isinstance(news_override, dict):
        if news_override.get("blackout"):
            return False, REASON_NEWS_BLACKOUT, str(news_override.get("reason") or "High-impact USD news blackout aktif")
        st_ov = str(news_override.get("status") or "READY").upper()
        age_ov = float(news_override.get("age_seconds") or 0.0)
        if block_on_stale and (st_ov in ("STALE", "UNAVAILABLE", "ERROR") or age_ov > max_stale_sec):
            return False, REASON_STALE_NEWS_DATA, str(news_override.get("reason") or f"Data news {st_ov} / stale")
        return True, REASON_OK, "News clear"

    if isinstance(snapshot_news, str) and snapshot_news.strip():
        low_n = snapshot_news.lower()
        if "gagal" in low_n or "stale" in low_n or "offline" in low_n:
            if block_on_stale:
                return False, REASON_STALE_NEWS_DATA, snapshot_news
        else:
            return False, REASON_NEWS_BLACKOUT, snapshot_news

    # 1. Cek US News Radar terverifikasi (non-blocking cache / state)
    radar_ready = False
    try:
        st_radar = us_news_radar.get_state(force=False, non_blocking=True)
        if isinstance(st_radar, dict):
            r_status = str(st_radar.get("status") or "UNAVAILABLE").upper()
            now_u = datetime.now(timezone.utc)
            if r_status == "READY":
                radar_ready = True
                for ev in (st_radar.get("events") or []):
                    if str(ev.get("verification") or "").upper() != "VERIFIED":
                        continue
                    if str(ev.get("impact") or "").upper() != "HIGH":
                        continue
                    dt_ev = datetime.fromisoformat(ev["dt_utc"])
                    diff_m = (dt_ev - now_u).total_seconds() / 60.0
                    if -blackout_min <= diff_m <= blackout_min:
                        return False, REASON_NEWS_BLACKOUT, f"High-impact news blackout: {ev.get('name')} ({diff_m:+.0f}m)"
            elif r_status == "STALE" and block_on_stale:
                return False, REASON_STALE_NEWS_DATA, "Cache US News Radar berstatus STALE"
    except Exception:
        radar_ready = False

    # 2. Cek kalender berita sekunder (`analisis_pasar.berita_blackout_aktif`)
    try:
        ff_blk = analisis_pasar.berita_blackout_aktif()
        if ff_blk:
            if "gagal" in ff_blk.lower():
                if block_on_stale and not radar_ready:
                    return False, REASON_STALE_NEWS_DATA, ff_blk
            else:
                return False, REASON_NEWS_BLACKOUT, f"High-impact news blackout: {ff_blk}"
        cache_ff = getattr(analisis_pasar, "_CACHE_BERITA", {})
        if isinstance(cache_ff, dict) and block_on_stale and not radar_ready:
            c_ts = float(cache_ff.get("waktu") or 0.0)
            c_fail = bool(cache_ff.get("gagal"))
            c_evs = cache_ff.get("events")
            if (c_fail and not c_evs) or (c_ts > 0 and (time.time() - c_ts) > max_stale_sec):
                return False, REASON_STALE_NEWS_DATA, "Data kalender berita USD tidak tersedia atau stale"
    except Exception:
        if block_on_stale and not radar_ready:
            return False, REASON_STALE_NEWS_DATA, "Verifikasi risiko berita gagal dijalankan"

    return True, REASON_OK, "News clear"


def evaluate_directional_alignment(
    direction: str,
    entry_mode: str,
    htf_bias: str,
    h1_bias: str,
    m30_status: str,
    has_explicit_htf_conflict: bool = False,
    allow_conflict_midzone: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Validasi arah struktur multi-timeframe per jalur entry (Section 4 & 6):
    - BUY membutuhkan bias BULLISH; SELL membutuhkan bias BEARISH.
    - EARLY_PENDING: H4 DAN H1 wajib mendukung arah entry (tanpa konflik HTF).
    - ADAPTIVE_ENTRY & CONFIRMED_ENTRY:
      * H4 wajib mendukung arah entry.
      * H1 neutral diperbolehkan apabila ALLOW_NEUTRAL_H1=true dan M30 mendukung arah entry
        (CONFIRMED / ALIGNED / BULLISH / BEARISH searah, bukan CONFLICT / WEAK / OPPOSING).
      * Konflik H4 vs H1 yang berlawanan langsung (mis. H4 BULLISH vs H1 BEARISH) tetap BLOCK (HTF_CONFLICT).
    """
    conf = cfg or get_auto_trading_config()
    if bool(conf.get("AUTO_LOOSE_MODE", False)):
        # LOOSE MODE: abaikan konflik HTF / bias arah (zona tetap wajib lolos validasi struktural & SL/TP)
        return True, REASON_OK, "LOOSE MODE: validasi arah HTF dilewati"
    dir_u = str(direction or "").upper()
    want_bias = "BULLISH" if dir_u == "BUY" else "BEARISH"
    opp_bias = "BEARISH" if dir_u == "BUY" else "BULLISH"

    h4_u = str(htf_bias or "NETRAL").upper()
    h1_u = str(h1_bias or "NETRAL").upper()
    m30_u = str(m30_status or "CONFIRMED").upper()
    mode_u = str(entry_mode or "ADAPTIVE_ENTRY").upper()

    # Normalisasi label H1
    h1_is_aligned = (want_bias in h1_u) or (h1_u == "ALIGNED")
    h1_is_opposing = (opp_bias in h1_u) or ("CONFLICT" in h1_u) or ("COUNTER" in h1_u)
    h1_is_neutral = (not h1_is_aligned) and (not h1_is_opposing)

    # Normalisasi label M30
    m30_is_opposing = any(tok in m30_u for tok in ("CONFLICT", "OPPOSING", "COUNTER", opp_bias))
    m30_is_supportive = (not m30_is_opposing) and any(
        tok in m30_u for tok in ("CONFIRMED", "ALIGNED", "STRONG", "SUPPORT", want_bias)
    )

    if allow_conflict_midzone and bool(conf.get("CONFLICT_MIDZONE_LIMIT_ENABLED", False)):
        return (
            True,
            REASON_OK,
            f"Mode Mid-Zone 50% Limit aktif saat market konflik/sideways (H4={h4_u}, H1={h1_u}, M30={m30_u})",
        )

    if h4_u == opp_bias or h1_is_opposing or has_explicit_htf_conflict:
        return (
            False,
            REASON_HTF_CONFLICT,
            f"Konflik struktur HTF signifikan (H4={h4_u}, H1={h1_u}, arah={dir_u})",
        )

    if h4_u != want_bias:
        return (
            False,
            REASON_DIRECTION_NOT_ALIGNED,
            f"Bias utama H4 ({h4_u}) belum mendukung arah {dir_u}",
        )

    if mode_u == "EARLY_PENDING":
        if not h1_is_aligned:
            return (
                False,
                REASON_HTF_CONFLICT if h1_is_opposing else REASON_DIRECTION_NOT_ALIGNED,
                f"EARLY_PENDING mewajibkan H4 dan H1 searah (H4={h4_u}, H1={h1_u})",
            )
        if m30_is_opposing:
            return (
                False,
                REASON_HTF_CONFLICT,
                f"EARLY_PENDING diblokir karena momentum M30 konflik ({m30_u})",
            )
        return True, REASON_OK, f"H4 ({h4_u}) & H1 ({h1_u}) searah mendukung EARLY_PENDING"

    # ADAPTIVE_ENTRY / CONFIRMED_ENTRY
    if h1_is_aligned:
        if m30_is_opposing:
            return False, REASON_HTF_CONFLICT, f"Momentum M30 konflik terhadap arah setup ({m30_u})"
        return True, REASON_OK, f"H4 ({h4_u}) & H1 ({h1_u}) mendukung {mode_u}"

    if h1_is_neutral and bool(conf.get("ALLOW_NEUTRAL_H1", True)):
        if m30_is_supportive:
            return True, REASON_OK, f"H4 ({h4_u}) & M30 ({m30_u}) mendukung {mode_u} (H1 netral diizinkan)"
        return (
            False,
            REASON_DIRECTION_NOT_ALIGNED,
            f"H1 netral memerlukan dukungan M30 searah (M30 saat ini: {m30_u})",
        )

    return False, REASON_DIRECTION_NOT_ALIGNED, f"Struktur H1 ({h1_u}) belum mendukung {dir_u}"


def validate_zone_quality_layer(
    zone_in: Any,
    frames: Optional[Dict[str, pd.DataFrame]],
    current_price: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    spread: float = 0.0,
    market_status: str = "OPEN",
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str, str]:
    """
    Validasi kualitas & keutuhan zona (Section 4 & 6):
    Mengembalikan (ok, state_if_failed, reason_code, reason_text).
    """
    conf = cfg or get_auto_trading_config()
    zone = _ensure_zone_obj(zone_in)
    min_score = float(conf.get("MIN_ZONE_SCORE", 70.0))

    z_status = str(zone.status or "ACTIVE").upper()
    z_fresh = str(zone.freshness or "FRESH").upper()
    z_touches = int(zone.touch_count or 0)

    if z_status == "EXPIRED" or z_fresh == "EXPIRED" or getattr(zone, "is_expired", False):
        return False, STATE_EXPIRED, REASON_ZONE_EXPIRED, f"Zona {zone.zone_id} sudah EXPIRED"

    if (
        not zone.is_valid
        or zone.is_broken
        or z_status in ("INVALIDATED", "BROKEN", "REJECTED")
        or z_fresh == "INVALIDATED"
    ):
        # Zona invalid/broken/rejected TIDAK PERNAH dieksekusi, termasuk pada LOOSE MODE.
        # Bedakan invalidasi struktural dari sekadar skor rendah
        if (not zone.is_broken) and z_fresh not in ("INVALIDATED", "MITIGATED") and float(zone.score or 0.0) < min_score and "invalid" not in str(zone.rejection_reason or "").lower():
            return (
                False,
                STATE_BLOCKED,
                REASON_ZONE_LOW_SCORE,
                f"Zona {zone.zone_id} tidak memenuhi kualitas minimum ({zone.rejection_reason or f'score={zone.score:.1f}'})",
            )
        else:
            return (
                False,
                STATE_INVALIDATED,
                REASON_ZONE_INVALIDATED,
                f"Zona {zone.zone_id} sudah INVALIDATED ({zone.rejection_reason or z_status})",
            )

    if z_fresh == "MITIGATED" or z_touches > 2:
        return (
            False,
            STATE_INVALIDATED,
            REASON_ZONE_MITIGATED,
            f"Zona {zone.zone_id} sudah termitigasi ({z_touches}x sentuhan)",
        )

    if frames:
        closed_inv, closed_rsn, _, _, _ = evaluate_smc_zone_closed_invalidation(zone, frames)
        if closed_inv:
            return False, STATE_INVALIDATED, REASON_ZONE_INVALIDATED, closed_rsn or f"Zona {zone.zone_id} ditembus candle close"

    live_inv, live_rsn, _, _, _ = evaluate_smc_setup_live_invalidation(
        zone, current_price=current_price, bid=bid, ask=ask,
        frames=frames, spread=spread, market_status=market_status,
    )
    if live_inv:
        return False, STATE_INVALIDATED, REASON_ZONE_INVALIDATED, live_rsn or f"Setup SL zona {zone.zone_id} ditembus harga live"

    # Cek tembus structural invalidation langsung pada harga live
    inval = float(zone.invalidation_price or 0.0)
    if inval > 0 and current_price > 0:
        if (zone.zone_type == "DEMAND" and current_price < inval) or (zone.zone_type == "SUPPLY" and current_price > inval):
            return (
                False,
                STATE_INVALIDATED,
                REASON_ZONE_INVALIDATED,
                f"Harga ({current_price:.2f}) telah menembus structural invalidation ({inval:.2f})",
            )

    if float(zone.score or 0.0) < min_score:
        return (
            False,
            STATE_BLOCKED,
            REASON_ZONE_LOW_SCORE,
            f"Score zona ({zone.score:.1f}) di bawah minimum {min_score:.1f}",
        )

    if is_far_reference_zone(zone, current_price=current_price) and not bool(conf.get("AUTO_LOOSE_MODE", False)):
        return (
            False,
            STATE_RADAR,
            REASON_ZONE_TOO_FAR,
            f"Zona {zone.zone_id} masih berstatus FAR_REFERENCE di luar jangkauan aktif",
        )

    return True, STATE_RADAR, REASON_OK, "Zona valid & aktif"


def _snr_zone_to_candidate_dict(snr_z: Dict[str, Any], symbol: str = SYMBOL) -> Optional[Dict[str, Any]]:
    """
    Konversi zona SNR aktif tervalidasi menjadi representasi kandidat zona seragam
    agar strategi hybrid SMC + SNR dapat dieksekusi oleh validator & execution engine
    tanpa melonggarkan standar validasi zona.
    """
    if not isinstance(snr_z, dict):
        return None
    zid = str(snr_z.get("zone_id") or "").strip()
    if not zid:
        return None
    z_type_raw = str(snr_z.get("zone_type") or "").upper()
    if z_type_raw in ("SUPPORT", "DEMAND"):
        z_type = "DEMAND"
    elif z_type_raw in ("RESISTANCE", "SUPPLY"):
        z_type = "SUPPLY"
    else:
        return None

    lo = float(snr_z.get("low") or snr_z.get("price_low") or 0.0)
    hi = float(snr_z.get("high") or snr_z.get("price_high") or 0.0)
    if lo <= 0 or hi < lo:
        return None

    life_st = str(snr_z.get("lifecycle_status") or snr_z.get("status") or "ACTIVE").upper()
    if life_st in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED", "REJECTED"):
        return None

    score = float(snr_z.get("score") or 0.0)
    grade = str(snr_z.get("grade") or ("A+" if score >= 85 else ("A" if score >= 75 else ("B" if score >= 70 else "C"))))
    tf = str(snr_z.get("timeframe") or "H1").upper()
    inval = float(snr_z.get("invalidation_price") or (lo if z_type == "DEMAND" else hi))
    touch_cnt = int(snr_z.get("touch_count") or 0)
    fresh = "FRESH" if touch_cnt == 0 else ("TESTED_1X" if touch_cnt == 1 else "TESTED_2X")

    return {
        "zone_id": zid,
        "symbol": symbol,
        "timeframe": tf,
        "zone_type": z_type,
        "price_low": round(lo, 2),
        "price_high": round(hi, 2),
        "midpoint": round((lo + hi) / 2.0, 2),
        "score": score,
        "grade": grade,
        "origin_bos": {"tipe": "SNR_STRUCTURE", "arah": "bull" if z_type == "DEMAND" else "bear", "origin_valid": True},
        "fvg_confirmation": None,
        "displacement_strength": float(snr_z.get("displacement_strength") or 1.5),
        "freshness": fresh,
        "touch_count": touch_cnt,
        "liquidity_sweep_status": "NONE",
        "invalidation_price": round(inval, 2),
        "is_valid": True,
        "rejection_reason": "",
        "bos_candle_time": snr_z.get("confirmed_time") or snr_z.get("formation_time"),
        "bos_timeframe": tf,
        "status": "ACTIVE",
        "source": "SNR",
        "risk_tier": str(snr_z.get("risk_tier") or ""),
        "zone_relevance": str(snr_z.get("zone_relevance") or "ACTIVE"),
        "distance": float(snr_z.get("distance_points") or snr_z.get("distance") or 0.0),
    }


def select_candidate_zones(
    scan_result: Optional[Dict[str, Any]],
    decision: Optional[Dict[str, Any]],
    snr_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Kumpulkan kandidat zona terbaik secara terurut dari hasil Python Scanner existing (Hybrid SMC + SNR):
    1. Fokus & zona aktif SMC (utama).
    2. Zona aktif SNR tervalidasi (jika tersedia / sebagai kandidat hybrid saat SMC tidak memiliki kandidat aktif).
    Tidak menduplikasi zona yang sama.
    """
    sc = scan_result or {}
    dec = decision or {}
    snr_r = snr_result or {}
    seen = set()
    ordered: List[Dict[str, Any]] = []

    def _add(z_item: Any):
        if z_item is None:
            return
        zd = z_item.to_dict() if hasattr(z_item, "to_dict") else (dict(z_item) if isinstance(z_item, dict) else None)
        if not isinstance(zd, dict):
            return
        zid = str(zd.get("zone_id") or "").strip()
        if not zid or zid in seen or str(zd.get("timeframe") or "").upper() == "ACTIVE_SIGNAL":
            return
        seen.add(zid)
        ordered.append(zd)

    _add(dec.get("zone"))
    _add(dec.get("primary_zone"))
    prio = dec.get("radar_priority") or sc.get("radar_priority") or {}
    _add(prio.get("primary_directional_zone"))
    _add(prio.get("best_confirmed_candidate"))
    _add(prio.get("best_pending_candidate"))

    act_list = sorted(
        [z for z in (sc.get("active_zones") or []) if isinstance(z, dict)],
        key=lambda z: (
            0 if str(z.get("direction_status") or "") == "INSIDE_ZONE" else 1,
            -float(z.get("score") or 0.0),
            float(z.get("distance") or 9999.0),
        ),
    )
    for z in act_list:
        _add(z)

    # Hybrid SMC + SNR: sertakan zona SNR aktif tervalidasi sebagai kandidat pengganti / pelengkap
    snr_scan = (
        snr_r.get("scan_result")
        if isinstance(snr_r.get("scan_result"), dict)
        else (snr_r.get("scan") if isinstance(snr_r.get("scan"), dict) else snr_r)
    )
    snr_dec = snr_r.get("decision") if isinstance(snr_r.get("decision"), dict) else {}
    sym_ref = str(sc.get("symbol") or snr_scan.get("symbol") or SYMBOL)
    live_px_sel = float(sc.get("current_price") or sc.get("mid") or 0.0)
    from modules.snr.entry_validator import _is_live_active_snr_zone

    def _snr_live_ok(sz_item: Any) -> bool:
        # Hanya zona SNR yang masih aktif secara lifecycle & valid terhadap harga live yang jadi kandidat eksekusi
        if not isinstance(sz_item, dict):
            return False
        zt = str(sz_item.get("zone_type") or "").upper()
        exp_t = "SUPPORT" if zt in ("SUPPORT", "DEMAND") else "RESISTANCE"
        return _is_live_active_snr_zone(sz_item, exp_t, live_px_sel)

    if isinstance(snr_dec.get("target_zone"), dict) and _snr_live_ok(snr_dec.get("target_zone")):
        _add(_snr_zone_to_candidate_dict(snr_dec.get("target_zone"), symbol=sym_ref))
    snr_ranked = snr_scan.get("ranked_zones") if isinstance(snr_scan.get("ranked_zones"), dict) else {}
    snr_zones_dict = snr_scan.get("zones") if isinstance(snr_scan.get("zones"), dict) else {}
    snr_cand_lists = [
        snr_scan.get("supports"),
        snr_scan.get("resistances"),
        snr_scan.get("active_zones"),
        snr_ranked.get("active_supports"),
        snr_ranked.get("active_resistances"),
        snr_zones_dict.get("active_supports"),
        snr_zones_dict.get("active_resistances"),
        snr_zones_dict.get("supports"),
        snr_zones_dict.get("resistances"),
    ]
    snr_pool_merged = []
    for lst in snr_cand_lists:
        for sz in (lst or []):
            if isinstance(sz, dict):
                snr_pool_merged.append(sz)
    snr_pool_merged.sort(
        key=lambda z: (
            -float(z.get("score") or 0.0),
            float(z.get("distance_points") or z.get("distance") or 9999.0),
        )
    )
    for sz in snr_pool_merged:
        if _snr_live_ok(sz):
            _add(_snr_zone_to_candidate_dict(sz, symbol=sym_ref))

    return ordered


def validate_trade_candidate(
    zone_in: Any,
    mt5_api: Any,
    state_mgr: ExecutionStateManager,
    snapshot: Optional[Dict[str, Any]] = None,
    scan_result: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
    trading_mode: Optional[str] = None,
    preferred_entry_mode: Optional[str] = None,
    news_override: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Jalankan 7 lapis validasi penuh pada 1 kandidat zona dan kembalikan hasil terstruktur
    yang konsisten pada seluruh mode trading (SCANNER_ONLY, DRY_RUN, SEMI_AUTO, FULL_AUTO).
    Analisis market & trade plan TIDAK berubah karena perbedaan trading_mode!
    """
    conf = cfg or get_auto_trading_config()
    snap = snapshot or {}
    sc = scan_result or {}
    dec = decision or {}
    zone = _ensure_zone_obj(zone_in)

    mode_info = state_mgr.get_mode_info()
    eff_trading_mode = str(trading_mode or mode_info.get("trading_mode") or conf.get("TRADING_MODE", "SCANNER_ONLY")).upper()
    eff_entry_pref = str(preferred_entry_mode or mode_info.get("entry_mode") or conf.get("ENTRY_MODE", "ADAPTIVE")).upper()
    symbol = str(sc.get("symbol") or snap.get("symbol") or zone.symbol or conf.get("SYMBOL") or SYMBOL)

    direction = "BUY" if zone.zone_type == "DEMAND" else "SELL"
    setup_id = make_setup_id(zone.zone_id, direction, symbol=symbol)

    frames = snap.get("frames") if isinstance(snap.get("frames"), dict) else None
    htf_ctx = sc.get("htf_context") or {}
    h4_bias = str(htf_ctx.get("H4") or sc.get("htf_bias") or "NETRAL").upper()
    h1_bias = str(htf_ctx.get("H1") or sc.get("h1_structure") or "NETRAL").upper()
    m30_status = str(
        ((dec.get("stages") or {}).get("M30") or {}).get("status")
        or htf_ctx.get("m30_status")
        or sc.get("m30_status")
        or "CONFIRMED"
    ).upper()
    has_htf_conflict = bool(htf_ctx.get("has_conflict"))

    # 1. Inspeksi Akun & Koneksi MT5
    acc_snap = inspect_mt5_account_and_permissions(
        mt5_api=mt5_api,
        symbol=symbol,
        state_mgr=state_mgr,
        require_exec_permission=(eff_trading_mode in ("FULL_AUTO", "SEMI_AUTO")),
        trading_mode=eff_trading_mode,
        cfg=conf,
    )
    tick = acc_snap.get("tick")
    bid = float(getattr(tick, "bid", 0.0) or 0.0) if tick is not None else float(snap.get("bid") or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0) if tick is not None else float(snap.get("ask") or 0.0)
    live_price = round((bid + ask) / 2.0, 2) if (bid > 0 and ask >= bid) else float(
        sc.get("current_price") or snap.get("current_price") or 0.0
    )
    spread = round(ask - bid, 3) if (bid > 0 and ask >= bid) else float(sc.get("spread") or snap.get("spread") or 0.0)
    mkt_status = str(sc.get("market_status") or snap.get("market_status") or ("OPEN" if acc_snap.get("market_open") else "CLOSED")).upper()
    if not acc_snap.get("market_open") and acc_snap.get("connected"):
        mkt_status = "CLOSED"

    # 2. Evaluasi Analisis Entry Path & Levels (Independen dari Trading Mode!)
    snr_ctx = dec.get("snr_context") or sc.get("snr_context")
    opposing_zones = [
        z for z in (sc.get("active_zones") or []) + (sc.get("radar_zones") or [])
        if isinstance(z, dict) and str(z.get("zone_type") or "").upper() != zone.zone_type
    ]
    existing_plan = dec.get("plan") if isinstance(dec.get("plan"), dict) and (dec["plan"].get("zone_id") == zone.zone_id) else None

    path_eval = evaluate_entry_paths_for_zone(
        zone,
        frames=frames,
        current_price=live_price,
        bid=bid if bid > 0 else None,
        ask=ask if ask > 0 else None,
        spread=spread,
        htf_bias=h4_bias,
        h1_bias=h1_bias,
        m30_status=m30_status,
        has_htf_conflict=has_htf_conflict,
        server_time=snap.get("server_time"),
        opposing_zones=opposing_zones,
        snr_context=snr_ctx,
        existing_plan=existing_plan,
        preferred_mode=eff_entry_pref,
        cfg=conf,
    )

    chosen_entry_mode = str(path_eval["entry_mode"])
    chosen_sub_mode = str(path_eval.get("sub_mode") or "")
    is_conflict_midzone = (chosen_sub_mode == "CONFLICT_MIDZONE_LIMIT")
    order_kind = str(path_eval["order_kind"])
    is_market_order = (order_kind == "MARKET")
    levels = path_eval["levels"]
    touch = path_eval["touch"]
    triggers = path_eval["triggers"]

    from modules.config import compute_zone_limit_entry_price
    planned_entry = float(
        touch["exec_price"]
        if is_market_order
        else (
            levels.get("entry_price")
            or compute_zone_limit_entry_price(
                zone.price_low,
                zone.price_high,
                demand=(direction == "BUY"),
                zone=zone,
                cfg=conf,
                digits=2,
            )
        )
    )
    sl_px = float(levels.get("sl") or 0.0)
    tp1_px = float(levels.get("tp1") or 0.0)
    tp2_px = float(levels.get("tp2") or 0.0)
    inval_px = float(levels.get("invalidation_price") or zone.invalidation_price or 0.0)

    # Susun objek Trade Plan murni (tetap tersedia untuk pre-market / SCANNER_ONLY)
    raw_risk = abs(planned_entry - sl_px) if (planned_entry > 0 and sl_px > 0) else 0.0
    raw_reward = abs(tp1_px - planned_entry) if (planned_entry > 0 and tp1_px > 0) else 0.0
    planned_rr = round(raw_reward / raw_risk, 2) if raw_risk > 0 else 0.0

    trade_plan = {
        "setup_id": setup_id,
        "zone_id": zone.zone_id,
        "symbol": symbol,
        "direction": direction,
        "order_type": path_eval["order_type"],
        "entry_mode": chosen_entry_mode,
        "sub_mode": chosen_sub_mode,
        "zone_timeframe": zone.timeframe,
        "zone_type": f"{zone.timeframe} {zone.zone_type}",
        "zone_low": zone.price_low,
        "zone_high": zone.price_high,
        "zone_score": zone.score,
        "zone_grade": zone.grade,
        "entry": round(planned_entry, 2),
        "sl": round(sl_px, 2),
        "tp": round(tp1_px, 2),
        "tp1": round(tp1_px, 2),
        "tp2": round(tp2_px, 2),
        "planned_rr": planned_rr,
        "invalidation_price": inval_px,
    }

    market_bias = {
        "direction": direction,
        "H4": h4_bias,
        "H1": h1_bias,
        "M30": m30_status,
        "M5": triggers["primary_trigger"] if triggers["any_valid"] else ("ZONE TOUCHED" if touch["touched"] else "WAITING"),
    }

    # Cek apakah setup ini memang SUDAH berstatus ORDER_PLACED / POSITION_ACTIVE pada broker
    existing_rec = state_mgr.get_setup(setup_id)
    if isinstance(existing_rec, dict) and existing_rec.get("state") in (STATE_POSITION_ACTIVE, STATE_ORDER_PLACED):
        cur_st = str(existing_rec["state"])
        return {
            "setup_id": setup_id,
            "zone_id": zone.zone_id,
            "symbol": symbol,
            "trading_mode": eff_trading_mode,
            "market_bias": market_bias,
            "trade_plan": trade_plan,
            "zone": zone.to_dict(),
            "zone_quality": existing_rec.get("zone_quality") or classify_zone_quality_label(zone, cfg=conf),
            "touch": touch,
            "triggers": triggers,
            "entry_mode": existing_rec.get("entry_mode") or chosen_entry_mode,
            "target_lot": float(existing_rec.get("target_lot") or 0.0),
            "final_lot": float(existing_rec.get("final_lot") or existing_rec.get("actual_lot") or 0.0),
            "lot_reason": "ACTIVE_ON_BROKER",
            "actual_rr": planned_rr,
            "risk_check": "PASSED",
            "validation": "PASSED",
            "entry_status": cur_st,
            "execution_status": cur_st,
            "decision": STATE_BLOCKED,   # Jangan kirim order baru karena setup sudah aktif!
            "state": cur_st,
            "reason_code": REASON_DUPLICATE_SETUP,
            "reason_text": f"Setup sudah aktif di broker ({cur_st})",
            "can_execute": False,
            "account": acc_snap,
            "live_price": live_price,
            "spread": spread,
        }

    def _resolve_next_action(st_v: str, r_code: str) -> str:
        if st_v == STATE_ENTRY_READY:
            return "Kirim order ke MT5" if eff_trading_mode == "FULL_AUTO" else (
                "Simulasi DRY_RUN" if eff_trading_mode == "DRY_RUN" else "Menunggu persetujuan / SCANNER_ONLY"
            )
        if st_v in (STATE_INVALIDATED, STATE_EXPIRED) or r_code == REASON_PRICE_LEFT_ENTRY_RANGE:
            return "Release plan invalid/missed & cari kandidat zona pengganti otomatis"
        if r_code == REASON_HTF_CONFLICT:
            return "Tahan eksekusi (monitoring conditional plan hingga konflik HTF selesai)"
        if st_v == STATE_WAIT_TRIGGER:
            return "Zona tersentuh — pantau trigger valid M5 pada candle close berikutnya"
        if st_v in (STATE_APPROACHING, STATE_RADAR):
            return "Pantau pergerakan harga mendekati zona aktif"
        return f"Periksa blocker ({r_code}) sebelum eksekusi"

    def _build_result(
        state_val: str,
        reason_code: str,
        reason_text: str,
        risk_chk: str = "NOT_CHECKED",
        val_chk: str = "FAILED",
        t_lot: float = 0.0,
        f_lot: float = 0.0,
        l_rsn: str = "",
        act_rr: float = planned_rr,
        z_qual: str = "",
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        qual_lbl = z_qual or classify_zone_quality_label(zone, strong_zone_validated=False, cfg=conf)
        can_exec = (state_val == STATE_ENTRY_READY)
        exec_st = "READY" if can_exec else ("PREMARKET_PLAN_ONLY" if reason_code == REASON_MARKET_CLOSED else state_val)
        res_d = {
            "setup_id": setup_id,
            "zone_id": zone.zone_id,
            "symbol": symbol,
            "trading_mode": eff_trading_mode,
            "market_bias": market_bias,
            "trade_plan": trade_plan,
            "zone": zone.to_dict(),
            "zone_quality": qual_lbl,
            "touch": touch,
            "triggers": triggers,
            "entry_mode": chosen_entry_mode,
            "sub_mode": chosen_sub_mode,
            "target_lot": t_lot,
            "final_lot": f_lot,
            "lot_reason": l_rsn or reason_code,
            "actual_rr": act_rr,
            "risk_check": risk_chk,
            "validation": val_chk,
            "entry_status": state_val,
            "execution_status": exec_st,
            "decision": state_val,
            "state": state_val,
            "reason_code": reason_code,
            "reason_text": reason_text,
            "next_action": _resolve_next_action(state_val, reason_code),
            "can_execute": can_exec,
            "account": acc_snap,
            "live_price": live_price,
            "spread": spread,
        }
        if isinstance(extra_fields, dict):
            res_d.update(extra_fields)
        return res_d

    # =========================================================================
    # LAYER 2: ZONE QUALITY VALIDATION (Evaluasi struktur zona terlebih dahulu)
    # =========================================================================
    z_ok, z_fail_state, z_code, z_msg = validate_zone_quality_layer(
        zone, frames=frames, current_price=live_price,
        bid=bid if bid > 0 else None, ask=ask if ask > 0 else None,
        spread=spread, market_status=mkt_status, cfg=conf,
    )
    if not z_ok:
        return _build_result(z_fail_state, z_code, z_msg, val_chk="BLOCKED" if z_fail_state != STATE_RADAR else "WAIT")

    # =========================================================================
    # LAYER 3: DIRECTIONAL VALIDATION & STRUCTURAL SNR BARRIER
    # =========================================================================
    dir_ok, dir_code, dir_msg = evaluate_directional_alignment(
        direction=direction,
        entry_mode=chosen_entry_mode,
        htf_bias=h4_bias,
        h1_bias=h1_bias,
        m30_status=m30_status,
        has_explicit_htf_conflict=has_htf_conflict,
        allow_conflict_midzone=is_conflict_midzone,
        cfg=conf,
    )
    if not dir_ok:
        return _build_result(STATE_BLOCKED, dir_code, dir_msg, val_chk="BLOCKED")

    # Evaluasi konflik SMC vs SNR secara struktural (hanya blokir pada mode STRUCTURAL; pada FIXED_RR 1:1 kita mengunci target ketat 1:1)
    if (
        isinstance(snr_ctx, dict)
        and planned_entry > 0
        and sl_px > 0
        and tp1_px > 0
        and not (str(conf.get("TP1_MODE", "FIXED_RR")).upper() in ("FIXED_RR", "FIXED", "1:1") and bool(conf.get("FORCE_FIXED_TP1", True)))
    ):
        barrier = evaluate_snr_opposing_barrier(
            direction, planned_entry, sl_px, tp1=tp1_px, current_price=live_price, snr_context=snr_ctx
        )
        if barrier.get("blocked"):
            clean_rr = float(barrier.get("clean_rr") or 0.0)
            if clean_rr < float(conf.get("MIN_RR", 1.5)) - 1e-9:
                return _build_result(
                    STATE_BLOCKED,
                    "SNR_STRUCTURAL_BARRIER",
                    str(barrier.get("reason") or "Opposing SNR zone memangkas ruang RR di bawah minimum"),
                    val_chk="BLOCKED",
                    act_rr=clean_rr,
                )

    # =========================================================================
    # LAYER 1: MANDATORY ENVIRONMENT & ACCOUNT HARD FILTERS
    # =========================================================================
    if state_mgr.get_mode_info().get("emergency_halt"):
        return _build_result(
            STATE_BLOCKED, "EMERGENCY_HALT",
            f"Emergency halt aktif: {state_mgr.get_mode_info().get('emergency_reason')}",
            val_chk="BLOCKED",
        )

    if not acc_snap.get("connected"):
        return _build_result(STATE_BLOCKED, REASON_MT5_DISCONNECTED, acc_snap.get("reason_text") or "MT5 Disconnected", val_chk="BLOCKED")

    if mkt_status == "CLOSED" or not acc_snap.get("market_open"):
        # Saat market closed: Trade Plan tetap dihasilkan utuh untuk pre-market analysis, tetapi eksekusi BLOCKED!
        return _build_result(
            STATE_BLOCKED,
            REASON_MARKET_CLOSED,
            "Market sedang tutup — menampilkan pre-market trade plan tanpa mengirim order",
            val_chk="PREMARKET_ONLY",
        )

    if acc_snap.get("reason_code") != REASON_OK:
        return _build_result(STATE_BLOCKED, acc_snap["reason_code"], acc_snap.get("reason_text", ""), val_chk="BLOCKED")

    # Cek kesegaran data tick & OHLC MT5
    fresh_status = str(sc.get("data_freshness") or snap.get("data_freshness") or "FRESH").upper()
    if fresh_status in ("STALE", "INVALID", "NO DATA", "NO_DATA"):
        return _build_result(
            STATE_BLOCKED,
            REASON_STALE_MARKET_DATA,
            f"Data market OHLC/tick tidak fresh ({fresh_status})",
            val_chk="BLOCKED",
        )

    max_tick_stale = float(conf.get("PANTAU_TICK_STALE_SECONDS", 30.0))
    if tick is not None and getattr(tick, "time", 0) > 0:
        # Bandingkan umur tick apabila tick.time adalah UTC epoch wajar
        tick_age_snap = snap.get("tick_age")
        if isinstance(tick_age_snap, (int, float)) and float(tick_age_snap) > max_tick_stale:
            return _build_result(
                STATE_BLOCKED,
                REASON_STALE_MARKET_DATA,
                f"Tick harga MT5 stale ({float(tick_age_snap):.1f}s > {max_tick_stale:.1f}s)",
                val_chk="BLOCKED",
            )

    # Cek Spread maksimum
    max_spread = float(conf.get("SPREAD_MAKS_POIN", 1.0))
    if spread > max_spread + 1e-9:
        return _build_result(
            STATE_BLOCKED,
            REASON_SPREAD_TOO_HIGH,
            f"Spread aktual ({spread:.2f} USD) melebihi batas {max_spread:.2f} USD",
            val_chk="BLOCKED",
        )

    # Cek News Blackout & Stale News Data
    news_ok, news_code, news_msg = evaluate_news_hard_filter(
        news_override=news_override,
        snapshot_news=snap.get("news"),
        cfg=conf,
    )
    if not news_ok:
        return _build_result(STATE_BLOCKED, news_code, news_msg, val_chk="BLOCKED")

    # Cek Duplicate Setup / Stopped-out Setup
    if conf.get("PREVENT_DUPLICATE_ORDER", True):
        is_dup, dup_code = state_mgr.is_setup_duplicate_or_locked(
            setup_id, zone_id=zone.zone_id, zone_bos_time=str(zone.bos_candle_time or "")
        )
        if is_dup:
            return _build_result(
                STATE_BLOCKED,
                dup_code,
                f"Proteksi anti-duplikat / anti-reentry aktif ({dup_code}) untuk {zone.zone_id}",
                val_chk="BLOCKED",
            )

    # =========================================================================
    # LAYER 4: ENTRY TRIGGER & EXECUTION RANGE VALIDATION
    # =========================================================================
    if not path_eval.get("trigger_passed"):
        p_code = str(path_eval.get("reason_code") or REASON_WAIT_TRIGGER)
        p_msg = str(path_eval.get("reason_text") or "")
        if p_code == REASON_PRICE_LEFT_ENTRY_RANGE:
            return _build_result(STATE_BLOCKED, REASON_PRICE_LEFT_ENTRY_RANGE, p_msg, val_chk="BLOCKED")
        if touch["touched"]:
            return _build_result(STATE_WAIT_TRIGGER, REASON_WAIT_TRIGGER, p_msg, val_chk="WAIT")
        if touch["is_approaching"]:
            return _build_result(STATE_APPROACHING, REASON_WAIT_ZONE_TOUCH, p_msg, val_chk="WAIT")
        return _build_result(STATE_RADAR, p_code, p_msg, val_chk="WAIT")

    # =========================================================================
    # LAYER 5: SUPPORTING CONFLUENCE (Informasional & Input Strong Zone, BUKAN Blocker)
    # =========================================================================
    supporting_confluence = {
        "fvg_status": (zone.fvg_confirmation or {}).get("status", "NONE") if isinstance(zone.fvg_confirmation, dict) else "NONE",
        "liquidity_sweep_origin": zone.liquidity_sweep_status,
        "displacement_atr": zone.displacement_strength,
        "new_fvg_required": not bool(conf.get("ALLOW_ENTRY_WITHOUT_NEW_FVG", True)),
    }

    # =========================================================================
    # LAYER 6 & 7: SL/TP, ACTUAL RR, RISK BUDGET, DYNAMIC LOT & BROKER EXECUTION CHECK
    # =========================================================================
    sl_tp_res = validate_sl_tp_and_rr(
        direction=direction,
        entry_price=planned_entry,
        sl_price=sl_px,
        tp1_price=tp1_px,
        tp2_price=tp2_px,
        structural_invalidation=inval_px,
        sym_info=acc_snap.get("symbol_info"),
        bid=bid if bid > 0 else None,
        ask=ask if ask > 0 else None,
        is_market_order=is_market_order,
        symbol=symbol,
        cfg=conf,
    )
    if not sl_tp_res.get("valid"):
        return _build_result(
            STATE_BLOCKED,
            str(sl_tp_res.get("reason_code")),
            str(sl_tp_res.get("reason_text")),
            risk_chk="FAILED",
            val_chk="BLOCKED",
            act_rr=float(sl_tp_res.get("actual_rr") or planned_rr),
        )

    actual_rr = float(sl_tp_res["actual_rr"])
    trade_plan.update({
        "entry": sl_tp_res["entry"],
        "sl": sl_tp_res["sl"],
        "tp": sl_tp_res["tp"],
        "tp1": sl_tp_res["tp1"],
        "tp2": sl_tp_res["tp2"],
        "actual_rr": actual_rr,
    })

    # Strong Zone Validation untuk kelayakan lot 0.05
    strong_val = validate_strong_zone(
        zone=zone,
        entry_mode=chosen_entry_mode,
        htf_aligned=dir_ok,
        has_htf_conflict=has_htf_conflict,
        actual_rr=actual_rr,
        trigger_valid=True,
        cfg=conf,
    )
    zone_quality_lbl = classify_zone_quality_label(zone, strong_zone_validated=strong_val["passed"], cfg=conf)

    # Portfolio Risk & Effective Risk Budget
    portfolio_risk = evaluate_portfolio_risk_and_daily_limits(
        mt5_api=mt5_api,
        state_mgr=state_mgr,
        symbol=symbol,
        cfg=conf,
    )
    risk_budget_info = compute_effective_risk_budget(
        entry_mode=chosen_entry_mode,
        portfolio_risk=portfolio_risk,
        cfg=conf,
        is_pending_order=(not is_market_order),
    )

    lot_res = resolve_dynamic_lot_for_trade(
        mt5_api=mt5_api,
        zone=zone,
        entry_mode=chosen_entry_mode,
        direction=direction,
        entry_price=sl_tp_res["entry"],
        sl_price=sl_tp_res["sl"],
        risk_budget_info=risk_budget_info,
        portfolio_risk=portfolio_risk,
        account_snapshot=acc_snap,
        strong_validation=strong_val,
        symbol=symbol,
        cfg=conf,
    )

    if not lot_res.get("allowed"):
        return _build_result(
            STATE_BLOCKED,
            str(lot_res.get("reason_code")),
            str(lot_res.get("reason_text")),
            risk_chk="FAILED",
            val_chk="BLOCKED",
            t_lot=float(lot_res.get("target_lot") or 0.0),
            f_lot=0.0,
            l_rsn=str(lot_res.get("lot_reason") or ""),
            act_rr=actual_rr,
            z_qual=zone_quality_lbl,
            extra_fields={
                "portfolio_risk": portfolio_risk,
                "risk_budget": risk_budget_info,
                "strong_validation": strong_val,
                "supporting_confluence": supporting_confluence,
            },
        )

    # Seluruh validasi 7 lapis TERPENUHI -> ENTRY_READY (tanpa WAIT tambahan!)
    return _build_result(
        STATE_ENTRY_READY,
        REASON_OK,
        f"Setup {chosen_entry_mode} tervalidasi penuh ({triggers['primary_trigger'] if is_market_order else 'LIMIT READY'})",
        risk_chk="PASSED",
        val_chk="PASSED",
        t_lot=float(lot_res["target_lot"]),
        f_lot=float(lot_res["final_lot"]),
        l_rsn=str(lot_res["lot_reason"]),
        act_rr=actual_rr,
        z_qual=zone_quality_lbl,
        extra_fields={
            "estimated_loss": float(lot_res["estimated_loss"]),
            "risk_percent": float(lot_res["risk_percent"]),
            "required_margin": float(lot_res["required_margin"]),
            "portfolio_risk": portfolio_risk,
            "risk_budget": risk_budget_info,
            "strong_validation": strong_val,
            "supporting_confluence": supporting_confluence,
            "sl_tp_validation": sl_tp_res,
        },
    )
