# -*- coding: utf-8 -*-
"""
zone_clustering.py — Sistem Pengelompokan (Clustering) Level & Kandidat Zona Support/Resistance
yang Berdekatan Berdasarkan Toleransi ATR dan Batas Lebar Maksimum Timeframe.
"""
from modules.snr.config import get_snr_config
from modules.snr.data_provider import safe_float


def _normalize_candidate_item(item, idx, zone_type, timeframe, default_atr, zone_cfg):
    """Menormalisasi input (harga float, dict swing, atau dict kandidat zona) menjadi kandidat standar."""
    tf = str(timeframe).upper()
    z_type = str(zone_type).upper()
    ref_atr = max(0.10, safe_float(default_atr, 2.0))
    min_w = max(0.15, ref_atr * safe_float(zone_cfg.get("min_zone_width_atr", 0.15), 0.15))

    if isinstance(item, (int, float)):
        p = round(float(item), 2)
        return {
            "candidate_id": f"{tf}_{z_type[:3]}_RAW_{idx}",
            "timeframe": tf,
            "zone_type": z_type,
            "low": p,
            "high": p,
            "mid": p,
            "width": 0.0,
            "ref_atr": ref_atr,
            "is_point_level": True,
            "source_swings": [{
                "pivot_id": f"{tf}_RAW_{idx}",
                "price": p,
                "bar_index": idx,
                "confirmed_index": idx,
                "swing_class": "MAJOR",
                "strength": 6.0,
            }],
            "formation_index": idx,
            "confirmed_index": idx,
            "swing_class": "MAJOR",
            "swing_strength": 6.0,
            "bounce_strength_atr": 1.0,
            "historical_reaction_count": 1,
            "repeated_rejections": 0,
            "consolidation_structure": "STANDARD_PIVOT",
            "age_bars": 10,
            "freshness_ratio": 0.85,
            "freshness_state": "ACTIVE_TESTED",
            "breakout_history": "CLEAN_HOLD",
        }

    if isinstance(item, dict) and "low" in item and "high" in item and "zone_type" in item:
        c = dict(item)
        c["low"] = round(safe_float(c["low"], 0.0), 2)
        c["high"] = round(safe_float(c["high"], c["low"]), 2)
        if c["high"] < c["low"]:
            c["low"], c["high"] = c["high"], c["low"]
        c["mid"] = round((c["low"] + c["high"]) / 2.0, 2)
        c["width"] = round(c["high"] - c["low"], 2)
        c["ref_atr"] = max(0.10, safe_float(c.get("ref_atr"), ref_atr))
        c["is_point_level"] = False
        return c

    if isinstance(item, dict):
        p = round(safe_float(item.get("price", item.get("low", item.get("high", 0.0))), 0.0), 2)
        low_p = round(safe_float(item.get("low", p), p), 2)
        high_p = round(safe_float(item.get("high", p), p), 2)
        if z_type == "SUPPORT" and high_p <= low_p:
            high_p = round(low_p + min_w, 2)
        elif z_type == "RESISTANCE" and high_p <= low_p:
            low_p = round(high_p - min_w, 2)
        return {
            "candidate_id": item.get("candidate_id", f"{tf}_{z_type[:3]}_SW_{idx}"),
            "timeframe": tf,
            "zone_type": z_type,
            "low": low_p,
            "high": high_p,
            "mid": round((low_p + high_p) / 2.0, 2),
            "width": round(high_p - low_p, 2),
            "ref_atr": max(0.10, safe_float(item.get("atr_at_pivot", default_atr), ref_atr)),
            "is_point_level": bool(item.get("low") is None and item.get("high") is None),
            "source_swings": [item],
            "formation_index": int(item.get("bar_index", idx)),
            "confirmed_index": int(item.get("confirmed_index", item.get("bar_index", idx))),
            "swing_class": item.get("swing_class", "MINOR"),
            "swing_strength": safe_float(item.get("strength", 5.0), 5.0),
            "bounce_strength_atr": safe_float(item.get("excursion_atr", 1.0), 1.0),
            "historical_reaction_count": int(item.get("historical_reaction_count", 0)),
            "repeated_rejections": int(item.get("repeated_rejections", 0)),
            "consolidation_structure": item.get("consolidation_structure", "STANDARD_PIVOT"),
            "age_bars": int(item.get("age_bars", 10)),
            "freshness_ratio": safe_float(item.get("freshness_ratio", 0.8), 0.8),
            "freshness_state": item.get("freshness_state", "FRESH_UNTESTED"),
            "breakout_history": item.get("breakout_history", "CLEAN_HOLD"),
        }

    return None


def _are_structurally_compatible(cluster_items, cand):
    """
    Memastikan dua kandidat tidak digabung jika secara struktural berbeda:
    - Tipe zona berbeda (SUPPORT vs RESISTANCE).
    - Status ditembus berbeda (zona yang masih CLEAN_HOLD tidak digabung dengan BROKEN).
    """
    if not cluster_items:
        return True
    first = cluster_items[0]
    if first.get("zone_type") != cand.get("zone_type"):
        return False

    bh_first = str(first.get("breakout_history", "CLEAN_HOLD"))
    bh_cand = str(cand.get("breakout_history", "CLEAN_HOLD"))
    is_broken_first = bh_first.startswith("BROKEN")
    is_broken_cand = bh_cand.startswith("BROKEN")
    if is_broken_first != is_broken_cand:
        return False
    return True


def _build_zone_from_cluster(cluster_items, timeframe, zone_type, zone_cfg):
    """Membentuk satu objek zona terklaster lengkap dengan ID stabil dan metadata audit."""
    tf = str(timeframe).upper()
    z_type = str(zone_type).upper()

    lows = [float(x["low"]) for x in cluster_items]
    highs = [float(x["high"]) for x in cluster_items]
    ref_atrs = [max(0.10, safe_float(x.get("ref_atr"), 1.0)) for x in cluster_items]
    avg_atr = sum(ref_atrs) / float(len(ref_atrs))

    z_low = round(min(lows), 2)
    z_high = round(max(highs), 2)

    # Jika seluruh input berupa titik harga (misal contoh 3914.20, 3915.10, 3916.00, 3915.60)
    # dan hanya ada 1 titik sehingga z_high == z_low, berikan lebar minimum dinamis berbasis ATR
    all_points = all(bool(x.get("is_point_level", False)) for x in cluster_items)
    if z_high <= z_low and not all_points:
        min_w = max(0.15, avg_atr * safe_float(zone_cfg.get("min_zone_width_atr", 0.15), 0.15))
        if z_type == "SUPPORT":
            z_high = round(z_low + min_w, 2)
        else:
            z_low = round(z_high - min_w, 2)
    elif z_high <= z_low and all_points and len(cluster_items) == 1:
        min_w = max(0.15, avg_atr * safe_float(zone_cfg.get("min_zone_width_atr", 0.15), 0.15))
        if z_type == "SUPPORT":
            z_high = round(z_low + min_w, 2)
        else:
            z_low = round(z_high - min_w, 2)

    z_mid = round((z_low + z_high) / 2.0, 2)
    z_width = round(z_high - z_low, 2)

    # Kumpulkan source swings unik (tanpa duplikasi bar_index pada timeframe yang sama)
    seen_swing_keys = set()
    source_swings = []
    for item in cluster_items:
        for sw in item.get("source_swings", []):
            sw_key = (sw.get("timeframe", tf), sw.get("bar_index"), round(safe_float(sw.get("price"), 0.0), 2))
            if sw_key not in seen_swing_keys:
                seen_swing_keys.add(sw_key)
                source_swings.append(sw)

    source_swings.sort(key=lambda s: int(s.get("bar_index", 0)))

    formation_index = min(int(x.get("formation_index", 0)) for x in cluster_items)
    confirmed_index = max(int(x.get("confirmed_index", formation_index)) for x in cluster_items)
    formation_time = cluster_items[0].get("formation_time")
    confirmed_time = cluster_items[-1].get("confirmed_time")

    has_major = any(x.get("swing_class") == "MAJOR" for x in cluster_items)
    swing_class = "MAJOR" if (has_major or len(source_swings) >= 2) else "MINOR"
    max_strength = max(safe_float(x.get("swing_strength"), 0.0) for x in cluster_items)
    cluster_bonus = min(2.0, 0.45 * max(0, len(source_swings) - 1))
    structural_strength = round(min(10.0, max_strength + cluster_bonus), 2)

    max_bounce_atr = max(safe_float(x.get("bounce_strength_atr"), 0.0) for x in cluster_items)
    hist_reactions = max(int(x.get("historical_reaction_count", 0)) for x in cluster_items) + max(0, len(source_swings) - 1)
    rep_rejections = max(int(x.get("repeated_rejections", 0)) for x in cluster_items)

    structures = [x.get("consolidation_structure", "STANDARD_PIVOT") for x in cluster_items]
    if "IMPULSIVE_DEPARTURE" in structures:
        consol_struct = "IMPULSIVE_DEPARTURE"
    elif "CONSOLIDATION_BASE" in structures:
        consol_struct = "CONSOLIDATION_BASE"
    else:
        consol_struct = "STANDARD_PIVOT"

    freshness_ratio = max(safe_float(x.get("freshness_ratio"), 0.5) for x in cluster_items)
    min_age_bars = min(int(x.get("age_bars", 0)) for x in cluster_items)
    breakout_history = cluster_items[0].get("breakout_history", "CLEAN_HOLD")
    is_broken = str(breakout_history).startswith("BROKEN")

    # Identitas zona stabil selama struktur pembentuknya tetap
    low_tag = int(round(z_low * 100))
    high_tag = int(round(z_high * 100))
    zone_id = f"SNR_{tf}_{z_type[:3]}_{formation_index}_{low_tag}_{high_tag}"

    return {
        "zone_id": zone_id,
        "timeframe": tf,
        "zone_type": z_type,
        "low": z_low,
        "high": z_high,
        "mid": z_mid,
        "width": z_width,
        "width_atr": round(z_width / max(0.10, avg_atr), 3),
        "ref_atr": round(avg_atr, 4),
        "source_swings": source_swings,
        "source_swing_count": len(source_swings),
        "clustered_candidate_count": len(cluster_items),
        "formation_index": int(formation_index),
        "formation_time": formation_time,
        "confirmed_index": int(confirmed_index),
        "confirmed_time": confirmed_time,
        "swing_class": swing_class,
        "swing_strength": structural_strength,
        "bounce_strength_atr": round(max_bounce_atr, 3),
        "historical_reaction_count": int(hist_reactions),
        "repeated_rejections": int(rep_rejections),
        "consolidation_structure": consol_struct,
        "age_bars": int(min_age_bars),
        "freshness_ratio": round(freshness_ratio, 3),
        "breakout_history": breakout_history,
        "is_active": not is_broken,
        "status": "BROKEN" if is_broken else "ACTIVE",
    }


def cluster_zones(
    candidates,
    timeframe="H1",
    zone_type="SUPPORT",
    atr=None,
    config=None,
):
    """
    Mengelompokkan kandidat level/zona yang berdekatan menjadi Zona Support/Resistance terklaster.
    Mematuhi batas toleransi ATR, batas lebar maksimum zona per timeframe, dan kompatibilitas struktural.
    """
    if not candidates:
        return []

    cfg = config or get_snr_config()
    tf = str(timeframe).upper()
    z_type = str(zone_type).upper()
    zone_cfg = cfg.get("zone", {}).get(tf, cfg.get("zone", {}).get("H1", {}))

    norm_list = []
    for idx, item in enumerate(candidates):
        it_type = item.get("zone_type", z_type) if isinstance(item, dict) else z_type
        norm = _normalize_candidate_item(item, idx, it_type, tf, atr, zone_cfg)
        if norm is not None:
            norm_list.append(norm)

    if not norm_list:
        return []

    # Urutkan berdasarkan batas bawah dan tengah harga
    norm_list.sort(key=lambda c: (c["low"], c["high"], c.get("formation_index", 0)))

    tol_mult = safe_float(zone_cfg.get("cluster_tolerance_atr", 0.32), 0.32)
    max_width_mult = safe_float(zone_cfg.get("max_zone_width_atr", 0.78), 0.78)
    max_width_pts = safe_float(zone_cfg.get("max_zone_width_points", 10.0), 10.0)

    clusters = []
    current_cluster = [norm_list[0]]

    for cand in norm_list[1:]:
        c_low = min(float(x["low"]) for x in current_cluster)
        c_high = max(float(x["high"]) for x in current_cluster)
        c_atr = sum(max(0.10, safe_float(x.get("ref_atr"), 1.0)) for x in current_cluster) / float(len(current_cluster))
        cand_atr = max(0.10, safe_float(cand.get("ref_atr"), c_atr))
        eff_atr = (c_atr + cand_atr) / 2.0

        tol_dist = eff_atr * tol_mult
        max_allowed_width = min(max_width_pts, max(0.35, eff_atr * max_width_mult))

        gap = float(cand["low"]) - c_high
        proposed_low = min(c_low, float(cand["low"]))
        proposed_high = max(c_high, float(cand["high"]))
        proposed_width = proposed_high - proposed_low

        if (
            _are_structurally_compatible(current_cluster, cand)
            and gap <= tol_dist
            and proposed_width <= max_allowed_width
        ):
            current_cluster.append(cand)
        else:
            clusters.append(current_cluster)
            current_cluster = [cand]

    if current_cluster:
        clusters.append(current_cluster)

    clustered_zones = [
        _build_zone_from_cluster(cl, tf, cl[0].get("zone_type", z_type), zone_cfg)
        for cl in clusters
    ]
    return clustered_zones

