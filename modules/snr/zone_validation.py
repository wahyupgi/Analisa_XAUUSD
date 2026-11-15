# -*- coding: utf-8 -*-
"""
zone_validation.py — Validasi Struktural, Lebar Zona, Freshness, dan Pemisahan Status Zona
(Active, Historical, Broken, Invalidated) untuk SNR Advanced Scanner.
"""
from modules.snr.config import get_snr_config
from modules.snr.data_provider import safe_float
from modules.snr.touch_analysis import analyze_zone_touches


def validate_single_zone(zone, df=None, current_price=None, config=None, has_unclosed_last_bar=False):
    """
    Memvalidasi satu zona Support/Resistance terhadap aturan lebar maksimum,
    validitas konfirmasi pivot, dan riwayat sentuhan/penembusan.
    """
    cfg = config or get_snr_config()
    tf = str(zone.get("timeframe", "H1")).upper()
    zone_cfg = cfg.get("zone", {}).get(tf, cfg.get("zone", {}).get("H1", {}))

    evaluated = dict(zone)
    if df is not None and "touch_analysis" not in evaluated:
        evaluated = analyze_zone_touches(evaluated, df, config=cfg, has_unclosed_last_bar=has_unclosed_last_bar)

    z_low = safe_float(evaluated.get("low"), 0.0)
    z_high = safe_float(evaluated.get("high"), z_low)
    z_width = round(max(0.0, z_high - z_low), 2)
    ref_atr = max(0.10, safe_float(evaluated.get("ref_atr"), 1.0))

    max_w_atr = safe_float(zone_cfg.get("max_zone_width_atr", 0.78), 0.78)
    max_w_pts = safe_float(zone_cfg.get("max_zone_width_points", 10.0), 10.0)
    allowed_max_width = min(max_w_pts, max(0.35, ref_atr * max_w_atr))

    reasons = []
    is_valid_structure = True

    if z_low <= 0 or z_high <= 0 or z_high < z_low:
        is_valid_structure = False
        reasons.append("Batas harga zona tidak valid (low/high <= 0 atau high < low).")

    if z_width > (allowed_max_width + 1e-4):
        is_valid_structure = False
        reasons.append(
            f"Lebar zona ({z_width:.2f}) melebihi batas maksimum timeframe {tf} ({allowed_max_width:.2f})."
        )

    swings = evaluated.get("source_swings", [])
    if swings and any(not bool(s.get("confirmed", True)) for s in swings):
        is_valid_structure = False
        reasons.append("Zona menggunakan pivot yang belum terkonfirmasi.")

    touch_status = str(evaluated.get("touch_status", "UNTESTED")).upper()
    is_broken = bool(evaluated.get("is_broken", touch_status == "BROKEN"))
    is_invalidated = bool(evaluated.get("is_invalidated", touch_status == "INVALIDATED"))

    if not is_valid_structure:
        lifecycle = "INVALIDATED"
    elif is_invalidated:
        lifecycle = "INVALIDATED"
        reasons.append("Zona telah ditembus jauh melewati batas invalidasi.")
    elif is_broken:
        lifecycle = "BROKEN"
        reasons.append("Zona telah ditembus (BROKEN); dialihkan ke pemantauan potensi retest/flip.")
    else:
        lifecycle = "ACTIVE"
        if current_price is not None and safe_float(current_price, 0.0) > 0:
            cp = safe_float(current_price, 0.0)
            z_type = str(evaluated.get("zone_type", "SUPPORT")).upper()
            dist_atr = abs(cp - safe_float(evaluated.get("mid"), cp)) / ref_atr
            # Jika harga sudah berada di sisi berlawanan dari fungsi asli zona
            if z_type == "SUPPORT" and cp < (z_low - 0.15 * ref_atr):
                lifecycle = "BROKEN"
                evaluated["is_broken"] = True
                evaluated["touch_status"] = "BROKEN"
                reasons.append("Harga saat ini berada tegas di bawah zona Support.")
            elif z_type == "RESISTANCE" and cp > (z_high + 0.15 * ref_atr):
                lifecycle = "BROKEN"
                evaluated["is_broken"] = True
                evaluated["touch_status"] = "BROKEN"
                reasons.append("Harga saat ini berada tegas di atas zona Resistance.")
            elif dist_atr > 12.0:
                lifecycle = "HISTORICAL"
                reasons.append(f"Zona masih utuh namun berjarak jauh ({dist_atr:.1f} ATR) sebagai level historis.")

    if not reasons:
        reasons.append("Zona valid dan aktif secara struktural.")

    evaluated["width"] = z_width
    evaluated["max_allowed_width"] = round(allowed_max_width, 2)
    evaluated["is_structurally_valid"] = bool(is_valid_structure)
    evaluated["lifecycle_status"] = lifecycle
    evaluated["is_active"] = lifecycle == "ACTIVE"
    evaluated["status"] = lifecycle
    evaluated["validation_reasons"] = reasons
    return evaluated


def validate_zones(zones, df=None, current_price=None, config=None, has_unclosed_last_bar=False):
    """
    Memvalidasi daftar zona dan mengelompokkannya ke dalam:
    - active_zones
    - historical_zones
    - broken_zones
    - invalidated_zones
    - all_zones
    """
    validated_all = []
    active_zones = []
    historical_zones = []
    broken_zones = []
    invalidated_zones = []

    for z in zones or []:
        vz = validate_single_zone(
            z,
            df=df,
            current_price=current_price,
            config=config,
            has_unclosed_last_bar=has_unclosed_last_bar,
        )
        validated_all.append(vz)
        st = vz.get("lifecycle_status", "ACTIVE")
        if st == "ACTIVE":
            active_zones.append(vz)
        elif st == "HISTORICAL":
            historical_zones.append(vz)
        elif st == "BROKEN":
            broken_zones.append(vz)
        else:
            invalidated_zones.append(vz)

    return {
        "all_zones": validated_all,
        "active_zones": active_zones,
        "historical_zones": historical_zones,
        "broken_zones": broken_zones,
        "invalidated_zones": invalidated_zones,
    }

