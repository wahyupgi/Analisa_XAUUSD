# -*- coding: utf-8 -*-
"""
zone_ranking.py — Sistem Kategori & Ranking Zona Support/Resistance untuk SNR Advanced Scanner.

Kategori Wajib:
1. STRONG SUPPORT
2. MODERATE SUPPORT
3. WEAK SUPPORT
4. STRONG RESISTANCE
5. MODERATE RESISTANCE
6. WEAK RESISTANCE

Memisahkan:
- Zona Aktif (Active Supports S1, S2... & Active Resistances R1, R2...)
- Zona Historis (Historical)
- Zona Broken (Broken / Role-Reversal Candidates)
- Zona Invalidated (Invalidated)
"""
from modules.snr.data_provider import safe_float


def classify_zone_strength_category(zone):
    """Menentukan kategori kekuatan zona (STRONG / MODERATE / WEAK SUPPORT/RESISTANCE)."""
    z_type = str(zone.get("zone_type", "SUPPORT")).upper()
    score = safe_float(zone.get("score"), 0.0)
    grade = str(zone.get("grade", "D")).upper()
    lifecycle = str(zone.get("lifecycle_status", zone.get("status", "ACTIVE"))).upper()
    has_conf = bool((zone.get("confluence_info") or {}).get("has_confluence", False))

    if lifecycle in ("INVALIDATED", "BROKEN") and str((zone.get("retest_info") or {}).get("status", "")) != "RETEST_CONFIRMED":
        tier = "WEAK"
    elif grade == "A" or (grade == "B" and has_conf and score >= 76.0):
        tier = "STRONG"
    elif grade in ("B", "C") and score >= 60.0:
        tier = "MODERATE"
    else:
        tier = "WEAK"

    full_category = f"{tier} {z_type}"
    return full_category, tier


def compute_zone_distance(zone, current_price):
    """Menghitung jarak zona dari harga saat ini (dalam poin dan satuan ATR)."""
    cp = safe_float(current_price, 0.0)
    z_low = safe_float(zone.get("low"), 0.0)
    z_high = safe_float(zone.get("high"), z_low)
    ref_atr = max(0.25, safe_float(zone.get("ref_atr"), 2.0))

    if cp <= 0:
        return 0.0, 0.0, True

    if z_low <= cp <= z_high:
        return 0.0, 0.0, True

    dist_pts = min(abs(cp - z_low), abs(cp - z_high))
    dist_atr = dist_pts / ref_atr
    return round(dist_pts, 2), round(dist_atr, 3), False


def _compute_ranking_priority(zone, current_price):
    """
    Menghitung skor prioritas ranking operasional (0-100) yang menyeimbangkan:
    - Score & Grade
    - Jarak dari harga saat ini (relevansi operasional)
    - Freshness
    - Multi-TF Confluence & H4 Major importance
    - Status Breakout & Retest
    """
    score = safe_float(zone.get("score"), 0.0)
    dist_pts, dist_atr, inside = compute_zone_distance(zone, current_price)
    fresh = safe_float(zone.get("freshness_ratio"), 0.7)
    has_conf = bool((zone.get("confluence_info") or {}).get("has_confluence", False))
    tf = str(zone.get("timeframe", "H1")).upper()
    sw_class = str(zone.get("swing_class", "MINOR")).upper()
    bo_status = str((zone.get("breakout_info") or {}).get("status", "NO_BREAKOUT")).upper()
    rt_status = str((zone.get("retest_info") or {}).get("status", "NO_RETEST")).upper()

    quality_part = score * 0.45
    proximity_part = 34.0 if inside else (32.0 / (1.0 + 0.38 * dist_atr))
    fresh_part = fresh * 7.0

    struct_bonus = 0.0
    if has_conf:
        struct_bonus += 5.0
    if tf == "H4" and sw_class == "MAJOR":
        struct_bonus += 4.5
    elif tf == "H1" and sw_class == "MAJOR":
        struct_bonus += 2.5

    if rt_status in ("RETEST_CONFIRMED", "RETEST_IN_PROGRESS"):
        struct_bonus += 5.0
    elif bo_status == "VALID_BREAKOUT":
        struct_bonus += 2.5

    return round(quality_part + proximity_part + fresh_part + struct_bonus, 2)


def _deduplicate_overlapping_ranked_zones(sorted_zones):
    """
    Menghindari pemilihan dua zona dari timeframe berbeda yang menumpuk pada level harga
    yang sama persis sebagai S1 & S2 (atau R1 & R2), sambil menyimpan zona dengan skor/prioritas tertinggi.
    """
    selected = []
    for z in sorted_zones:
        z_low = safe_float(z.get("low"), 0.0)
        z_high = safe_float(z.get("high"), z_low)
        z_mid = safe_float(z.get("mid"), (z_low + z_high) / 2.0)
        z_atr = max(0.25, safe_float(z.get("ref_atr"), 2.0))

        is_dup = False
        for ex in selected:
            e_low = safe_float(ex.get("low"), 0.0)
            e_high = safe_float(ex.get("high"), e_low)
            e_mid = safe_float(ex.get("mid"), (e_low + e_high) / 2.0)
            ov = min(z_high, e_high) - max(z_low, e_low)
            min_w = max(0.15, min(z_high - z_low, e_high - e_low))
            if ov > (0.50 * min_w) or abs(z_mid - e_mid) <= (0.20 * z_atr):
                is_dup = True
                break
        if not is_dup:
            selected.append(z)
    return selected


def rank_zones(zones, current_price=None):
    """
    Mengklasifikasikan dan meranking seluruh zona SNR menjadi:
    - active_supports (S1, S2, ...)
    - active_resistances (R1, R2, ...)
    - major_zones
    - historical_zones
    - broken_zones
    - invalidated_zones
    - by_category (6 kategori kekuatan wajib)
    """
    cp = safe_float(current_price, 0.0)

    by_category = {
        "STRONG SUPPORT": [],
        "MODERATE SUPPORT": [],
        "WEAK SUPPORT": [],
        "STRONG RESISTANCE": [],
        "MODERATE RESISTANCE": [],
        "WEAK RESISTANCE": [],
    }

    enriched_all = []
    active_sup_candidates = []
    active_res_candidates = []
    historical_zones = []
    broken_zones = []
    invalidated_zones = []
    major_zones = []

    for z in zones or []:
        item = dict(z)
        cat_full, cat_short = classify_zone_strength_category(item)
        dist_pts, dist_atr, inside = compute_zone_distance(item, cp)
        priority = _compute_ranking_priority(item, cp)

        item["strength_category"] = cat_full
        item["strength_short"] = cat_short
        item["distance_points"] = dist_pts
        item["distance_atr"] = dist_atr
        item["price_inside_zone"] = inside
        item["ranking_priority"] = priority

        if cat_full in by_category:
            by_category[cat_full].append(item)

        lifecycle = str(item.get("lifecycle_status", item.get("status", "ACTIVE"))).upper()
        z_type = str(item.get("zone_type", "SUPPORT")).upper()

        if (
            str(item.get("timeframe", "")).upper() == "H4"
            and str(item.get("swing_class", "")).upper() == "MAJOR"
            and lifecycle in ("ACTIVE", "HISTORICAL")
        ):
            major_zones.append(item)

        if lifecycle == "ACTIVE":
            if z_type == "SUPPORT":
                # Support aktif berada di bawah atau sedang diuji oleh harga saat ini
                if cp <= 0 or safe_float(item.get("low"), 0.0) <= (cp + 0.25 * max(0.5, safe_float(item.get("ref_atr"), 2.0))):
                    active_sup_candidates.append(item)
                else:
                    broken_zones.append(item)
            else:
                # Resistance aktif berada di atas atau sedang diuji oleh harga saat ini
                if cp <= 0 or safe_float(item.get("high"), 0.0) >= (cp - 0.25 * max(0.5, safe_float(item.get("ref_atr"), 2.0))):
                    active_res_candidates.append(item)
                else:
                    broken_zones.append(item)
        elif lifecycle == "HISTORICAL":
            historical_zones.append(item)
        elif lifecycle == "BROKEN":
            broken_zones.append(item)
        else:
            invalidated_zones.append(item)

        enriched_all.append(item)

    # Urutkan kandidat aktif berdasarkan ranking_priority tertinggi, lalu deduplikasi overlap lintas TF
    active_sup_candidates.sort(key=lambda x: (-x["ranking_priority"], x["distance_points"], -x.get("score", 0.0)))
    active_res_candidates.sort(key=lambda x: (-x["ranking_priority"], x["distance_points"], -x.get("score", 0.0)))

    dedup_sup = _deduplicate_overlapping_ranked_zones(active_sup_candidates)
    dedup_res = _deduplicate_overlapping_ranked_zones(active_res_candidates)

    # Untuk penyajian S1, S2 (terdekat ke bawah dari harga saat ini di antara zona relevan teratas)
    top_supports = sorted(dedup_sup[:6], key=lambda x: (x["distance_points"], -x["ranking_priority"]))
    top_resistances = sorted(dedup_res[:6], key=lambda x: (x["distance_points"], -x["ranking_priority"]))

    for idx_s, sz in enumerate(top_supports):
        sz["rank_label"] = f"S{idx_s + 1}"
    for idx_r, rz in enumerate(top_resistances):
        rz["rank_label"] = f"R{idx_r + 1}"

    broken_zones.sort(key=lambda x: (x["distance_points"], -x.get("score", 0.0)))
    historical_zones.sort(key=lambda x: (-x.get("score", 0.0), x["distance_points"]))
    major_zones.sort(key=lambda x: (-x.get("score", 0.0), x["distance_points"]))

    for k_cat in by_category:
        by_category[k_cat].sort(key=lambda x: (-x["ranking_priority"], -x.get("score", 0.0)))

    return {
        "all_zones": enriched_all,
        "active_supports": top_supports,
        "active_resistances": top_resistances,
        "major_zones": major_zones,
        "historical_zones": historical_zones,
        "broken_zones": broken_zones,
        "invalidated_zones": invalidated_zones,
        "by_category": by_category,
    }

