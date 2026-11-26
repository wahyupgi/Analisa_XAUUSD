# -*- coding: utf-8 -*-
"""
modules/dynamic_lot_manager.py
Dynamic Lot Management berbasis kualitas zona tervalidasi & risk budget akun aktual (IDC / USC / USD):
- Strong Zone Validation (Section 8) untuk kelayakan lot 0.05.
- Pemilihan Target Lot maksimum (0.01 / 0.02 / 0.05) tanpa manipulasi skor (Section 7 & 9).
- Perhitungan kerugian SL aktual dengan mt5.order_calc_profit() + buffer komisi & slippage
  tanpa hardcode konversi mata uang USD ke IDC.
- Step-down otomatis 0.05 -> 0.02 -> 0.01 -> BLOCK (MIN_LOT_EXCEEDS_RISK) + validasi volume & margin broker.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

from modules.config import SYMBOL, get_auto_trading_config, is_grade_a_wait_confirm_zone
from modules.execution_state import (
    REASON_INSUFFICIENT_MARGIN,
    REASON_INVALID_VOLUME,
    REASON_MIN_LOT_EXCEEDS_RISK,
    REASON_OK,
    REASON_ZONE_LOW_SCORE,
)


def _z_attr(zone: Any, key: str, default: Any = None) -> Any:
    if isinstance(zone, dict):
        return zone.get(key, default)
    return getattr(zone, key, default)


def _zone_timeframes_set(zone: Any) -> set:
    tf_raw = str(_z_attr(zone, "timeframe", "") or "").upper()
    parts = {p.strip() for p in tf_raw.replace("/", "+").split("+") if p.strip()}
    return parts


def has_htf_supply_demand_origin(zone: Any) -> Tuple[bool, bool]:
    """
    Cek apakah zona berasal dari Supply/Demand H4 atau H1, serta apakah memiliki confluence H4+H1.
    Zona murni M30, M15, M5 mengembalikan (False, False).
    """
    z_type = str(_z_attr(zone, "zone_type", "") or "").upper()
    if z_type not in ("DEMAND", "SUPPLY"):
        return False, False
    tfs = _zone_timeframes_set(zone)
    has_h4 = "H4" in tfs
    has_h1 = "H1" in tfs
    return (has_h4 or has_h1), (has_h4 and has_h1)


def is_ltf_only_zone(zone: Any) -> bool:
    """True apabila zona hanya berasal dari M30, M15, atau M5 tanpa komponen H4/H1."""
    tfs = _zone_timeframes_set(zone)
    if not tfs:
        return True
    return not any(tf in ("H4", "H1", "D1", "W1") for tf in tfs)


def validate_strong_zone(
    zone: Any,
    entry_mode: str,
    htf_aligned: bool = True,
    has_htf_conflict: bool = False,
    actual_rr: float = 0.0,
    trigger_valid: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Strong Zone Validation (Section 8) — syarat mutlak untuk membuka batas maksimum lot 0.05:
    1. Berasal dari Supply/Demand H4 atau H1 (Confluence H4+H1 mendapat prioritas utama).
    2. Score minimal STRONG_ZONE_SCORE (default 90).
    3. Zona masih ACTIVE dan valid (bukan INVALIDATED / EXPIRED / BROKEN).
    4. Memenuhi aturan freshness & mitigation (FRESH / TESTED_1X / TESTED_2X, touch <= 2).
    5. Structural invalidation jelas (> 0).
    6. Terdapat displacement atau BOS yang mendukung pembentukan zona.
    7. Struktur HTF mendukung arah transaksi & tidak terdapat konflik HTF signifikan.
    8. RR aktual minimal MIN_RR (default 1.5).
    9. Jalur entry adalah ADAPTIVE_ENTRY atau CONFIRMED_ENTRY dengan trigger valid (bukan EARLY_PENDING).
    """
    conf = cfg or get_auto_trading_config()
    strong_min_score = float(conf.get("STRONG_ZONE_SCORE", 90.0))
    min_rr = float(conf.get("MIN_RR", 1.5))
    min_disp = float(conf.get("STRONG_ZONE_MIN_DISPLACEMENT_ATR", 1.5))

    reasons = []
    has_htf_sd, has_h4_h1_conf = has_htf_supply_demand_origin(zone)
    score = float(_z_attr(zone, "score", 0.0) or 0.0)
    is_valid = bool(_z_attr(zone, "is_valid", False))
    is_broken = bool(_z_attr(zone, "is_broken", False))
    status = str(_z_attr(zone, "status", "ACTIVE") or "ACTIVE").upper()
    freshness = str(_z_attr(zone, "freshness", "FRESH") or "FRESH").upper()
    touch_count = int(_z_attr(zone, "touch_count", 0) or 0)
    inval_px = float(_z_attr(zone, "invalidation_price", 0.0) or 0.0)
    disp = float(_z_attr(zone, "displacement_strength", 0.0) or 0.0)
    origin_bos = _z_attr(zone, "origin_bos", None)
    has_bos_or_disp = (disp >= min_disp) or (isinstance(origin_bos, dict) and bool(origin_bos.get("tipe")) and disp >= 1.2)
    mode_u = str(entry_mode or "").upper()

    if not has_htf_sd:
        reasons.append("Zona bukan berasal dari Supply/Demand H4 atau H1")
    if score < strong_min_score:
        reasons.append(f"Score zona ({score:.1f}) < {strong_min_score:.1f}")
    if not is_valid or is_broken or status in ("INVALIDATED", "EXPIRED", "MITIGATED", "REJECTED", "BROKEN"):
        reasons.append(f"Status zona tidak ACTIVE ({status})")
    if freshness in ("INVALIDATED", "MITIGATED", "EXPIRED") or touch_count > 2:
        reasons.append(f"Freshness/mitigation tidak memenuhi syarat ({freshness}, touch={touch_count})")
    if inval_px <= 0:
        reasons.append("Structural invalidation zona tidak jelas")
    if not has_bos_or_disp:
        reasons.append(f"Displacement/BOS pembentuk zona kurang kuat (disp={disp:.2f} ATR)")
    if not htf_aligned or has_htf_conflict:
        reasons.append("Struktur HTF belum searah atau terdapat konflik HTF")
    if actual_rr < min_rr - 1e-9:
        reasons.append(f"RR aktual ({actual_rr:.2f}) < minimum 1:{min_rr:.2f}")
    if mode_u == "EARLY_PENDING":
        reasons.append("EARLY_PENDING dibatasi maksimal 0.01 lot")
    elif mode_u not in ("ADAPTIVE", "ADAPTIVE_ENTRY", "CONFIRMED", "CONFIRMED_ENTRY") or not trigger_valid:
        reasons.append("Trigger ADAPTIVE_ENTRY / CONFIRMED_ENTRY belum tervalidasi")

    passed = len(reasons) == 0
    return {
        "passed": passed,
        "has_htf_supply_demand": has_htf_sd,
        "has_h4_h1_confluence": has_h4_h1_conf,
        "reasons": reasons,
        "reason_summary": "STRONG_ZONE_VALIDATED" if passed else "; ".join(reasons),
    }


def classify_zone_quality_label(zone: Any, strong_zone_validated: bool = False, cfg: Optional[Dict[str, Any]] = None) -> str:
    """Klasifikasi label kualitas zona (INVALID / STANDARD / MEDIUM / STRONG)."""
    conf = cfg or get_auto_trading_config()
    if not bool(_z_attr(zone, "is_valid", False)):
        return "INVALID"
    score = float(_z_attr(zone, "score", 0.0) or 0.0)
    min_s = float(conf.get("MIN_ZONE_SCORE", 70.0))
    med_s = float(conf.get("MEDIUM_ZONE_SCORE", 80.0))
    str_s = float(conf.get("STRONG_ZONE_SCORE", 90.0))
    if score < min_s:
        return "LOW_QUALITY"
    if score >= str_s and strong_zone_validated:
        return "STRONG"
    if score >= med_s or is_grade_a_wait_confirm_zone(zone, cfg=conf):
        return "MEDIUM"
    return "STANDARD"


def select_target_lot(
    zone: Any,
    entry_mode: str,
    strong_zone_validated: Optional[bool] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Pemilihan target lot maksimum berdasarkan kualitas zona & jalur entry (Section 9):
    - Zona invalid atau score < 70 -> 0.0 (BLOCK)
    - EARLY_PENDING -> maksimal 0.01
    - Zona M30/M15/M5 -> maksimal 0.02
    - Score >= 90 + HTF Supply/Demand + Strong Zone Validation lolos -> 0.05
    - Score >= 80 (atau >= 90 tanpa Strong Zone Validation) -> 0.02
    - Score 70–79 -> 0.01
    Tidak pernah melebihi MAX_LOT (0.05).
    """
    conf = cfg or get_auto_trading_config()
    loose = bool(conf.get("AUTO_LOOSE_MODE", False))
    is_valid = bool(_z_attr(zone, "is_valid", False)) and not bool(_z_attr(zone, "is_broken", False))
    status = str(_z_attr(zone, "status", "ACTIVE") or "ACTIVE").upper()
    if not is_valid or status in ("INVALIDATED", "EXPIRED", "MITIGATED", "REJECTED", "BROKEN"):
        return 0.0

    score = float(_z_attr(zone, "score", 0.0) or 0.0)
    min_score = float(conf.get("MIN_ZONE_SCORE", 70.0))
    med_score = float(conf.get("MEDIUM_ZONE_SCORE", 80.0))
    strong_score = float(conf.get("STRONG_ZONE_SCORE", 90.0))

    lot_std = float(conf.get("LOT_STANDARD", 0.01))
    lot_med = float(conf.get("LOT_MEDIUM", 0.02))
    lot_str = float(conf.get("LOT_STRONG", 0.05))
    max_lot = min(0.05, float(conf.get("MAX_LOT", 0.05)))
    ep_max = min(max_lot, float(conf.get("EARLY_PENDING_MAX_LOT", 0.01)))
    ltf_max = min(max_lot, float(conf.get("LTF_MAX_LOT", 0.02)))
    htf_max = min(max_lot, float(conf.get("HTF_STRONG_MAX_LOT", 0.05)))

    if score < min_score:
        return 0.0

    mode_u = str(entry_mode or "").upper()
    if mode_u == "EARLY_PENDING":
        return round(min(lot_std, ep_max), 2)

    has_htf_sd, _ = has_htf_supply_demand_origin(zone)
    if strong_zone_validated is None:
        strong_zone_validated = bool(_z_attr(zone, "strong_zone_validated", False))

    if score >= strong_score and has_htf_sd and strong_zone_validated and not is_ltf_only_zone(zone):
        target = min(lot_str, htf_max)
    elif score >= med_score or is_grade_a_wait_confirm_zone(zone, cfg=conf):
        target = lot_med
    else:
        target = lot_std

    if is_ltf_only_zone(zone):
        target = min(target, ltf_max)

    if loose:
        # LOOSE MODE: grade di bawah B (C/D/dst) selalu 0.01 lot
        grade_u = str(_z_attr(zone, "grade", "") or "").strip().upper()
        if (grade_u and grade_u[0] not in ("A", "B")) or (
            not grade_u and score < min_score and not is_grade_a_wait_confirm_zone(zone, cfg=conf)
        ):
            target = min(target, float(conf.get("LOOSE_LOT", 0.01)))

    return round(min(target, max_lot), 2)


def select_target_lot_with_reason(
    zone: Any,
    entry_mode: str,
    strong_validation: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """Kembalikan (target_lot, target_lot_reason) secara eksplisit untuk audit & UI Pantau."""
    conf = cfg or get_auto_trading_config()
    score = float(_z_attr(zone, "score", 0.0) or 0.0)
    strong_ok = bool((strong_validation or {}).get("passed")) if isinstance(strong_validation, dict) else bool(_z_attr(zone, "strong_zone_validated", False))
    target = select_target_lot(zone, entry_mode, strong_zone_validated=strong_ok, cfg=conf)
    if target <= 0.0:
        return 0.0, REASON_ZONE_LOW_SCORE

    mode_u = str(entry_mode or "").upper()
    strong_score = float(conf.get("STRONG_ZONE_SCORE", 90.0))
    if mode_u == "EARLY_PENDING":
        return target, "EARLY_PENDING_MAX_LOT_CAP"
    if score >= strong_score and is_ltf_only_zone(zone):
        return target, "LTF_MAX_LOT_CAP"
    if score >= strong_score and not strong_ok:
        return target, "STRONG_ZONE_VALIDATION_CAP"
    return target, "ZONE_QUALITY_MATCH"


def build_mt5_loss_calculator(
    mt5_api: Any,
    symbol: str,
    direction: str,
    entry_price: float,
    sl_price: float,
    sym_info: Any = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Callable[[float], Optional[float]]:
    """
    Buat fungsi loss_calculator(lot) -> estimated_loss_in_account_currency.
    Menggunakan mt5.order_calc_profit() secara langsung pada akun aktif (IDC / USC / USD)
    TANPA konversi hardcoded USD ke IDC, ditambah buffer slippage & estimasi komisi proporsional.
    """
    conf = cfg or get_auto_trading_config()
    dir_u = str(direction or "").upper()
    is_buy = (dir_u == "BUY")
    order_type = getattr(mt5_api, "ORDER_TYPE_BUY", 0) if is_buy else getattr(mt5_api, "ORDER_TYPE_SELL", 1)
    slippage_pts = max(0.0, float(conf.get("SLIPPAGE_BUFFER_PRICE", 0.20)))
    comm_usd_per_lot = max(0.0, float(conf.get("COMMISSION_PER_LOT_USD_EST", 7.0)))
    contract_size = float(getattr(sym_info, "trade_contract_size", 100.0) or 100.0) if sym_info is not None else 100.0

    # Harga SL efektif dengan slippage buffer konservatif
    worst_sl = (float(sl_price) - slippage_pts) if is_buy else (float(sl_price) + slippage_pts)

    def _calc(lot: float) -> Optional[float]:
        if mt5_api is None or lot <= 0 or entry_price <= 0 or sl_price <= 0:
            return None
        try:
            raw_pnl = mt5_api.order_calc_profit(order_type, symbol, float(lot), float(entry_price), float(worst_sl))
        except Exception:
            raw_pnl = None

        if raw_pnl is None or not math.isfinite(float(raw_pnl)):
            return None
        # Untuk arah SL yang valid, raw_pnl bernilai negatif
        if float(raw_pnl) >= 0:
            return None

        base_loss = abs(float(raw_pnl))

        # Konversi komisi ke mata uang akun secara dinamis lewat rasio nilai 1.0 USD pergerakan harga
        comm_in_acc_cur = 0.0
        if comm_usd_per_lot > 0 and contract_size > 0:
            try:
                ref_close = (float(entry_price) - 1.0) if is_buy else (float(entry_price) + 1.0)
                one_pt_pnl = mt5_api.order_calc_profit(order_type, symbol, float(lot), float(entry_price), ref_close)
                if one_pt_pnl is not None and math.isfinite(float(one_pt_pnl)) and float(one_pt_pnl) != 0:
                    # 1.00 USD price move pada XAUUSD (100 oz) = contract_size * lot USD
                    # Maka 1 USD bernilai: abs(one_pt_pnl) / (contract_size * lot) dalam mata uang akun!
                    acc_per_usd = abs(float(one_pt_pnl)) / (contract_size * float(lot))
                    comm_in_acc_cur = comm_usd_per_lot * float(lot) * acc_per_usd
            except Exception:
                comm_in_acc_cur = 0.0

        return round(base_loss + comm_in_acc_cur, 4)

    return _calc


def is_broker_volume_valid(lot: float, sym_info: Any) -> Tuple[bool, str]:
    """Validasi lot terhadap volume_min, volume_max, dan kelipatan volume_step spesifikasi simbol broker."""
    if lot <= 0 or not math.isfinite(lot):
        return False, "Lot <= 0"
    if sym_info is None:
        return True, "OK"
    vol_min = float(getattr(sym_info, "volume_min", 0.01) or 0.01)
    vol_max = float(getattr(sym_info, "volume_max", 100.0) or 100.0)
    vol_step = float(getattr(sym_info, "volume_step", 0.01) or 0.01)

    if lot < vol_min - 1e-9:
        return False, f"Lot {lot:.2f} < volume_min broker {vol_min:.2f}"
    if lot > vol_max + 1e-9:
        return False, f"Lot {lot:.2f} > volume_max broker {vol_max:.2f}"
    if vol_step > 0:
        steps = lot / vol_step
        if abs(steps - round(steps)) > 1e-6:
            return False, f"Lot {lot:.4f} tidak sesuai volume_step broker {vol_step:g}"
    return True, "OK"


def select_final_lot(
    target_lot: float,
    risk_budget: float,
    loss_calculator: Callable[[float], Optional[float]],
) -> float:
    """
    Fungsi inti pemilihan final lot (Section 9):
    Mencoba tier (0.05, 0.02, 0.01) dari yang terbesar <= target_lot yang muat dalam risk_budget.
    Mengembalikan 0.0 jika bahkan 0.01 melebihi risk_budget.
    """
    if target_lot <= 0 or risk_budget <= 0:
        return 0.0
    for lot in (0.05, 0.02, 0.01):
        if lot > target_lot + 1e-9:
            continue
        estimated_loss = loss_calculator(lot)
        if estimated_loss is None:
            continue
        if estimated_loss <= 0:
            continue
        if estimated_loss <= risk_budget + 1e-9:
            return lot
    return 0.0


def resolve_dynamic_lot_for_trade(
    mt5_api: Any,
    zone: Any,
    entry_mode: str,
    direction: str,
    entry_price: float,
    sl_price: float,
    risk_budget_info: Dict[str, Any],
    portfolio_risk: Dict[str, Any],
    account_snapshot: Dict[str, Any],
    strong_validation: Optional[Dict[str, Any]] = None,
    symbol: str = SYMBOL,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Pipeline lengkap pemilihan Dynamic Lot (Target Lot -> Risk Budget Downgrade -> Broker Volume -> Margin Check):
    - Jika target_lot == 0.0 -> BLOCK (ZONE_LOW_SCORE)
    - Jika 0.05 > risk_budget -> downgrade ke 0.02
    - Jika 0.02 > risk_budget -> downgrade ke 0.01
    - Jika 0.01 > risk_budget -> BLOCK (MIN_LOT_EXCEEDS_RISK)
    - Validasi volume_min / volume_max / volume_step broker (INVALID_VOLUME)
    - Validasi margin bebas dengan mt5.order_calc_margin() (INSUFFICIENT_MARGIN)
    """
    conf = cfg or get_auto_trading_config()
    sym_info = account_snapshot.get("symbol_info")
    free_margin = float(account_snapshot.get("free_margin") or 0.0)
    equity = float(account_snapshot.get("equity") or portfolio_risk.get("equity") or 0.0)
    currency = str(account_snapshot.get("currency") or portfolio_risk.get("currency") or "")

    target_lot, target_reason = select_target_lot_with_reason(
        zone, entry_mode, strong_validation=strong_validation, cfg=conf
    )
    if target_lot <= 0.0:
        return {
            "allowed": False,
            "target_lot": 0.0,
            "final_lot": 0.0,
            "estimated_loss": 0.0,
            "risk_percent": 0.0,
            "required_margin": 0.0,
            "currency": currency,
            "lot_reason": REASON_ZONE_LOW_SCORE,
            "reason_code": REASON_ZONE_LOW_SCORE,
            "reason_text": f"Kualitas/skor zona di bawah minimum ({float(_z_attr(zone, 'score', 0.0)):.1f} < {conf.get('MIN_ZONE_SCORE', 70.0)})",
        }

    if not risk_budget_info.get("allowed", False):
        r_code = str(risk_budget_info.get("reason_code") or REASON_MIN_LOT_EXCEEDS_RISK)
        return {
            "allowed": False,
            "target_lot": target_lot,
            "final_lot": 0.0,
            "estimated_loss": 0.0,
            "risk_percent": 0.0,
            "required_margin": 0.0,
            "currency": currency,
            "lot_reason": r_code,
            "reason_code": r_code,
            "reason_text": str(risk_budget_info.get("reason_text") or r_code),
        }

    risk_budget = float(risk_budget_info.get("risk_budget_nominal") or 0.0)
    loss_calc = build_mt5_loss_calculator(
        mt5_api=mt5_api,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        sl_price=sl_price,
        sym_info=sym_info,
        cfg=conf,
    )

    dir_u = str(direction or "").upper()
    ord_type = getattr(mt5_api, "ORDER_TYPE_BUY", 0) if dir_u == "BUY" else getattr(mt5_api, "ORDER_TYPE_SELL", 1)

    tiers = (
        float(conf.get("LOT_STRONG", 0.05)),
        float(conf.get("LOT_MEDIUM", 0.02)),
        float(conf.get("LOT_STANDARD", 0.01)),
    )

    any_vol_invalid = False
    vol_err_msg = ""
    any_margin_insufficient = False
    any_risk_exceeded = False
    min_tier_loss = None

    for lot in tiers:
        if lot > target_lot + 1e-9:
            continue

        vol_ok, vol_msg = is_broker_volume_valid(lot, sym_info)
        if not vol_ok:
            any_vol_invalid = True
            vol_err_msg = vol_msg
            continue

        est_loss = loss_calc(lot)
        if est_loss is None or est_loss <= 0:
            continue
        min_tier_loss = est_loss

        if est_loss > risk_budget + 1e-9:
            any_risk_exceeded = True
            continue

        # Cek margin yang dibutuhkan menggunakan mt5.order_calc_margin()
        req_margin = 0.0
        if mt5_api is not None and hasattr(mt5_api, "order_calc_margin"):
            try:
                m_val = mt5_api.order_calc_margin(ord_type, symbol, float(lot), float(entry_price))
                if m_val is not None and math.isfinite(float(m_val)):
                    req_margin = float(m_val)
            except Exception:
                req_margin = 0.0

        if req_margin > 0 and free_margin > 0 and req_margin > free_margin:
            any_margin_insufficient = True
            continue

        risk_pct = round((est_loss / equity * 100.0), 4) if equity > 0 else 0.0
        if lot < target_lot - 1e-9:
            lot_reason = "MARGIN_ADJUSTMENT" if any_margin_insufficient and not any_risk_exceeded else "RISK_LIMIT_ADJUSTMENT"
        else:
            lot_reason = target_reason

        return {
            "allowed": True,
            "target_lot": target_lot,
            "final_lot": round(lot, 2),
            "estimated_loss": round(est_loss, 4),
            "risk_percent": risk_pct,
            "required_margin": round(req_margin, 2),
            "currency": currency,
            "lot_reason": lot_reason,
            "reason_code": REASON_OK,
            "reason_text": f"Final lot {lot:.2f} ({lot_reason})",
        }

    # Gagal menemukan lot valid -> tentukan reason code spesifik
    if any_vol_invalid and not any_risk_exceeded and not any_margin_insufficient:
        return {
            "allowed": False,
            "target_lot": target_lot,
            "final_lot": 0.0,
            "estimated_loss": 0.0,
            "risk_percent": 0.0,
            "required_margin": 0.0,
            "currency": currency,
            "lot_reason": REASON_INVALID_VOLUME,
            "reason_code": REASON_INVALID_VOLUME,
            "reason_text": vol_err_msg or "Volume lot tidak sesuai spesifikasi simbol broker",
        }

    if any_margin_insufficient and not any_risk_exceeded:
        return {
            "allowed": False,
            "target_lot": target_lot,
            "final_lot": 0.0,
            "estimated_loss": 0.0,
            "risk_percent": 0.0,
            "required_margin": 0.0,
            "currency": currency,
            "lot_reason": REASON_INSUFFICIENT_MARGIN,
            "reason_code": REASON_INSUFFICIENT_MARGIN,
            "reason_text": f"Free margin ({free_margin:,.2f} {currency}) tidak mencukupi untuk minimal lot",
        }

    loss_txt = f"{min_tier_loss:,.2f} {currency}" if min_tier_loss is not None else "N/A"
    return {
        "allowed": False,
        "target_lot": target_lot,
        "final_lot": 0.0,
        "estimated_loss": round(min_tier_loss or 0.0, 4),
        "risk_percent": round(((min_tier_loss or 0.0) / equity * 100.0), 4) if equity > 0 else 0.0,
        "required_margin": 0.0,
        "currency": currency,
        "lot_reason": REASON_MIN_LOT_EXCEEDS_RISK,
        "reason_code": REASON_MIN_LOT_EXCEEDS_RISK,
        "reason_text": (
            f"Estimasi rugi lot minimum 0.01 ({loss_txt}) melebihi risk budget "
            f"({risk_budget:,.2f} {currency})"
        ),
    }
