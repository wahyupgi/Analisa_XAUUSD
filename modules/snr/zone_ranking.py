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
from modules.snr.breakout_detection import classify_snr_breakout_state
from modules.snr.data_provider import safe_float

try:
    from modules.config import ZONE_RELEVANCE_TF_LIMITS
except Exception:
    ZONE_RELEVANCE_TF_LIMITS = {
        "M5": (1.8, 3.5, 3.5, 18.0, 7.0, 30.0),
        "M15": (2.5, 5.0, 5.0, 25.0, 10.0, 40.0),
        "M30": (3.2, 6.5, 6.5, 32.0, 13.0, 50.0),
        "H1": (4.0, 8.0, 8.0, 40.0, 16.0, 60.0),
        "H4": (4.5, 10.0, 10.0, 45.0, 20.0, 75.0),
        "D1": (4.5, 10.0, 10.0, 45.0, 20.0, 75.0),
        "W1": (4.5, 10.0, 10.0, 45.0, 20.0, 75.0),
    }


def classify_snr_zone_relevance(zone, current_price=None, h1_atr=None):
    """
    Klasifikasi konsisten relevansi zona SNR: ACTIVE / RADAR / FAR_REFERENCE / INVALID.
    Zona terlalu jauh menurut ATR/timeframe ditandai FAR_REFERENCE (tetap valid sebagai referensi,
    bukan Active Focus dan tidak menghasilkan entry plan baru).
    """
    if not isinstance(zone, dict):
        return "INVALID", "Zona tidak valid"

    lifecycle = str(zone.get("lifecycle_status", zone.get("status", "ACTIVE"))).upper()
    bo_info = zone.get("breakout_info") if isinstance(zone.get("breakout_info"), dict) else {}
    bo_status = str(bo_info.get("status", "NO_BREAKOUT")).upper()
    if (
        zone.get("is_invalidated")
        or zone.get("is_broken")
        or zone.get("is_valid") is False
        or bo_status == "VALID_BREAKOUT"
        or bo_info.get("zone_failed")
        or bo_info.get("reclaim_detected")
        or lifecycle in ("INVALIDATED", "BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED", "EXPIRED", "MITIGATED")
    ):
        return "INVALID", f"Zona {lifecycle} / broken"

    cp = safe_float(current_price, 0.0)
    dist_pts, dist_atr, inside = compute_zone_distance(zone, cp) if cp > 0 else (
        safe_float(zone.get("distance_points"), 0.0),
        safe_float(zone.get("distance_atr"), 0.0),
        bool(zone.get("price_inside_zone")),
    )
    tf = str(zone.get("timeframe") or "H1").upper()
    a_atr, r_atr, a_min_pt, a_max_pt, r_min_pt, r_max_pt = ZONE_RELEVANCE_TF_LIMITS.get(
        tf, (4.0, 8.0, 8.0, 40.0, 16.0, 60.0)
    )
    eff_atr = safe_float(h1_atr, 0.0)
    if eff_atr <= 0:
        eff_atr = max(0.25, safe_float(zone.get("ref_atr"), 2.0))
    active_pt = max(a_min_pt, min(a_max_pt, a_atr * eff_atr))
    radar_pt = max(r_min_pt, min(r_max_pt, r_atr * eff_atr))

    if lifecycle == "HISTORICAL" or zone.get("is_distant_historical") or dist_atr > 12.0 or (dist_pts > radar_pt and not inside):
        return "FAR_REFERENCE", f"Jarak {dist_pts:.1f} USD ({dist_atr:.1f}x ATR) melebihi batas radar {tf} ({radar_pt:.1f} USD)"
    if inside or dist_pts <= active_pt:
        return "ACTIVE", f"Dalam jangkauan aktif {tf} ({dist_pts:.1f} USD <= {active_pt:.1f} USD)"
    return "RADAR", f"Dalam jangkauan radar {tf} ({dist_pts:.1f} USD <= {radar_pt:.1f} USD)"


def classify_zone_strength_category(zone):
    """Menentukan kategori kekuatan zona (STRONG / MODERATE / WEAK SUPPORT/RESISTANCE)."""
    z_type = str(zone.get("zone_type", "SUPPORT")).upper()
    score = safe_float(zone.get("score"), 0.0)
    grade = str(zone.get("grade", "D")).upper()
    lifecycle = str(zone.get("lifecycle_status", zone.get("status", "ACTIVE"))).upper()
    has_conf = bool((zone.get("confluence_info") or {}).get("has_confluence", False))

    if lifecycle in (
        "INVALIDATED", "BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED",
        "BREAKDOWN_PENDING", "BREAKOUT_PENDING", "UNCONFIRMED",
    ) and str((zone.get("retest_info") or {}).get("status", "")) != "RETEST_CONFIRMED":
        tier = "WEAK"
    elif grade == "A" or (grade == "B" and has_conf and score >= 76.0):
        tier = "STRONG"
    elif grade in ("B", "C") and score >= 60.0:
        tier = "MODERATE"
    else:
        tier = "WEAK"

    full_category = f"{tier} {z_type}"
    return full_category, tier


def compute_zone_distance(zone, current_price, digits=None, spec=None):
    """Menghitung jarak zona dari harga saat ini (dalam Price Distance USD/oz dan satuan ATR)."""
    from modules.utils import calc_price_distance, get_symbol_unit_spec

    cp = safe_float(current_price, 0.0)
    z_low = safe_float(zone.get("low"), 0.0)
    z_high = safe_float(zone.get("high"), z_low)
    ref_atr = max(0.25, safe_float(zone.get("ref_atr"), 2.0))

    if cp <= 0:
        return 0.0, 0.0, True

    dist_usd = calc_price_distance(z_low, z_high, cp)
    if dist_usd == 0.0:
        return 0.0, 0.0, True

    d_use = int(digits) if digits is not None else int(get_symbol_unit_spec(spec=spec)["digits"])
    dist_atr = dist_usd / ref_atr
    return round(dist_usd, d_use), round(dist_atr, 3), False


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


def _is_qualified_ab_zone(zone) -> bool:
    grd = str((zone or {}).get("grade") or "").strip().upper()
    sc = safe_float((zone or {}).get("score"), 0.0)
    return bool(grd.startswith(("A", "B")) or sc >= 70.0)


def _deduplicate_overlapping_ranked_zones(sorted_zones):
    """
    Menghindari pemilihan dua zona dari timeframe berbeda yang menumpuk pada level harga
    yang sama persis sebagai S1 & S2 (atau R1 & R2), sambil menyimpan zona dengan skor/prioritas tertinggi.
    Juga mencegah zona bayangan Grade rendah (< B / < 70) yang menempel tipis di tepi zona kuat Grade A/B
    ikut muncul dan menutupi zona utama.
    """
    selected = []
    for z in sorted_zones:
        z_low = safe_float(z.get("low"), 0.0)
        z_high = safe_float(z.get("high"), z_low)
        z_mid = safe_float(z.get("mid"), (z_low + z_high) / 2.0)
        z_atr = max(0.25, safe_float(z.get("ref_atr"), 2.0))
        z_ab = _is_qualified_ab_zone(z)

        is_dup = False
        for ex in selected:
            e_low = safe_float(ex.get("low"), 0.0)
            e_high = safe_float(ex.get("high"), e_low)
            e_mid = safe_float(ex.get("mid"), (e_low + e_high) / 2.0)
            e_atr = max(0.25, safe_float(ex.get("ref_atr"), 2.0))
            ov = min(z_high, e_high) - max(z_low, e_low)
            min_w = max(0.15, min(z_high - z_low, e_high - e_low))
            if ov > (0.50 * min_w) or abs(z_mid - e_mid) <= (0.20 * z_atr):
                is_dup = True
                break
            # Jika zona terpilih sudah Grade A/B dan kandidat baru hanya Grade C/D yang menempel tipis di tepinya
            # (mis. pantulan front-run 0.12 USD di depan H1 Support), serap ke zona kuat agar tidak menggeser S1/R1
            if _is_qualified_ab_zone(ex) and (not z_ab):
                edge_gap = max(0.0, max(z_low, e_low) - min(z_high, e_high))
                if edge_gap <= max(1.50, 0.35 * max(z_atr, e_atr)):
                    is_dup = True
                    break
        if not is_dup:
            selected.append(z)
    return selected


def rank_zones(zones, current_price=None, h1_atr=None):
    """
    Mengklasifikasikan dan meranking seluruh zona SNR menjadi:
    - active_supports (S1, S2, ...)
    - active_resistances (R1, R2, ...)
    - far_reference_zones (zona valid namun di luar jangkauan ATR/TF)
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
    far_reference_zones = []
    historical_zones = []
    broken_zones = []
    invalidated_zones = []
    unconfirmed_zones = []
    pending_break_zones = []
    reclaimed_zones = []
    major_zones = []

    from modules.utils import get_symbol_unit_spec, price_distance_to_mt5_points
    u_spec = get_symbol_unit_spec()
    for z in zones or []:
        item = dict(z)
        cat_full, cat_short = classify_zone_strength_category(item)
        dist_pts, dist_atr, inside = compute_zone_distance(item, cp, spec=u_spec)
        priority = _compute_ranking_priority(item, cp)

        item["strength_category"] = cat_full
        item["strength_short"] = cat_short
        item["distance_points"] = dist_pts
        item["distance_usd"] = dist_pts
        item["distance_mt5_points"] = price_distance_to_mt5_points(dist_pts, spec=u_spec)
        item["distance_atr"] = dist_atr
        item["price_inside_zone"] = inside
        item["ranking_priority"] = priority

        if cat_full in by_category:
            by_category[cat_full].append(item)

        lifecycle = str(item.get("lifecycle_status", item.get("status", "ACTIVE"))).upper()
        z_type = str(item.get("zone_type", "SUPPORT")).upper()
        bo_info = item.get("breakout_info") if isinstance(item.get("breakout_info"), dict) else {}
        bo_status = str(bo_info.get("status", "NO_BREAKOUT")).upper()
        zone_failed = bool(
            item.get("is_broken")
            or item.get("is_invalidated")
            or bo_status == "VALID_BREAKOUT"
            or bo_info.get("zone_failed")
            or bo_info.get("reclaim_detected")
        )
        z_rel, z_rel_reason = classify_snr_zone_relevance(item, cp, h1_atr=h1_atr)
        item["zone_relevance"] = z_rel
        item["zone_relevance_reason"] = z_rel_reason
        item["is_far_reference"] = bool(z_rel == "FAR_REFERENCE")

        if (
            str(item.get("timeframe", "")).upper() == "H4"
            and str(item.get("swing_class", "")).upper() == "MAJOR"
            and lifecycle in ("ACTIVE", "HISTORICAL")
            and not zone_failed
        ):
            major_zones.append(item)

        if lifecycle in ("ACTIVE", "TOUCHED", "SWEEP"):
            if item.get("is_invalidated"):
                item["lifecycle_status"] = "INVALIDATED"
                item["status"] = "INVALIDATED"
                item["is_active"] = False
                item["zone_relevance"] = "INVALID"
                item["is_far_reference"] = False
                invalidated_zones.append(item)
            elif zone_failed:
                item["lifecycle_status"] = "RECLAIMED" if bo_info.get("reclaim_detected") else "BROKEN"
                item["status"] = item["lifecycle_status"]
                item["is_broken"] = True
                item["is_active"] = False
                item["zone_relevance"] = "INVALID"
                item["is_far_reference"] = False
                broken_zones.append(item)
                if item["lifecycle_status"] == "RECLAIMED":
                    reclaimed_zones.append(item)
            elif bo_status == "BREAKOUT_PENDING_CONFIRMATION":
                pend_st = "BREAKDOWN_PENDING" if z_type == "SUPPORT" else "BREAKOUT_PENDING"
                item["lifecycle_status"] = pend_st
                item["status"] = pend_st
                item["is_active"] = False
                unconfirmed_zones.append(item)
                pending_break_zones.append(item)
            elif z_type == "SUPPORT":
                # Support aktif berada di bawah atau sedang diuji oleh harga saat ini (cp >= z_low)
                if cp <= 0 or safe_float(item.get("low"), 0.0) <= cp:
                    if z_rel == "FAR_REFERENCE":
                        # Terlalu jauh: tetap valid sebagai FAR REFERENCE, bukan Active Focus / entry
                        item["is_active"] = False
                        item["is_far_reference"] = True
                        far_reference_zones.append(item)
                        historical_zones.append(item)
                    else:
                        active_sup_candidates.append(item)
                else:
                    item["lifecycle_status"] = "BREAKDOWN_PENDING"
                    item["status"] = "BREAKDOWN_PENDING"
                    item["is_active"] = False
                    unconfirmed_zones.append(item)
                    pending_break_zones.append(item)
            else:
                # Resistance aktif berada di atas atau sedang diuji oleh harga saat ini (cp <= z_high)
                if cp <= 0 or safe_float(item.get("high"), 0.0) >= cp:
                    if z_rel == "FAR_REFERENCE":
                        item["is_active"] = False
                        item["is_far_reference"] = True
                        far_reference_zones.append(item)
                        historical_zones.append(item)
                    else:
                        active_res_candidates.append(item)
                else:
                    item["lifecycle_status"] = "BREAKOUT_PENDING"
                    item["status"] = "BREAKOUT_PENDING"
                    item["is_active"] = False
                    unconfirmed_zones.append(item)
                    pending_break_zones.append(item)
        elif lifecycle == "HISTORICAL":
            item["is_active"] = False
            if not zone_failed:
                item["zone_relevance"] = "FAR_REFERENCE"
                item["is_far_reference"] = True
                far_reference_zones.append(item)
            historical_zones.append(item)
        elif lifecycle in ("BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED"):
            item["is_active"] = False
            item["zone_relevance"] = "INVALID"
            item["is_far_reference"] = False
            item["is_distant_historical"] = bool(dist_atr > 12.0)
            broken_zones.append(item)
            if lifecycle in ("RECLAIM_PENDING", "RECLAIMED"):
                reclaimed_zones.append(item)
        elif lifecycle in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING", "UNCONFIRMED"):
            item["is_active"] = False
            if item.get("is_invalidated"):
                item["lifecycle_status"] = "INVALIDATED"
                item["status"] = "INVALIDATED"
                item["zone_relevance"] = "INVALID"
                item["is_far_reference"] = False
                invalidated_zones.append(item)
            elif zone_failed:
                item["lifecycle_status"] = "RECLAIMED" if bo_info.get("reclaim_detected") else "BROKEN"
                item["status"] = item["lifecycle_status"]
                item["is_broken"] = True
                item["zone_relevance"] = "INVALID"
                item["is_far_reference"] = False
                item["is_distant_historical"] = bool(dist_atr > 12.0)
                broken_zones.append(item)
                if item["lifecycle_status"] == "RECLAIMED":
                    reclaimed_zones.append(item)
            else:
                expected_pend = "BREAKDOWN_PENDING" if z_type == "SUPPORT" else "BREAKOUT_PENDING"
                on_valid_side = (
                    cp > 0
                    and (
                        (z_type == "SUPPORT" and cp > safe_float(item.get("high"), 0.0))
                        or (z_type == "RESISTANCE" and cp < safe_float(item.get("low"), 0.0))
                    )
                )
                if lifecycle in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING") and lifecycle != expected_pend:
                    # Arah pending bertentangan dengan tipe zona (mis. BREAKDOWN_PENDING pada RESISTANCE)
                    if (dist_atr > 12.0 or z_rel == "FAR_REFERENCE") and on_valid_side:
                        item["lifecycle_status"] = "HISTORICAL"
                        item["status"] = "HISTORICAL"
                        item["zone_relevance"] = "FAR_REFERENCE"
                        item["is_far_reference"] = True
                        far_reference_zones.append(item)
                        historical_zones.append(item)
                    else:
                        item["lifecycle_status"] = "UNCONFIRMED"
                        item["status"] = "UNCONFIRMED"
                        unconfirmed_zones.append(item)
                elif lifecycle in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING") and (dist_atr > 12.0 or z_rel == "FAR_REFERENCE") and on_valid_side and bo_status == "NO_BREAKOUT":
                    # Event lama yang jauh di sisi valid bukan pending break aktif
                    item["lifecycle_status"] = "HISTORICAL"
                    item["status"] = "HISTORICAL"
                    item["zone_relevance"] = "FAR_REFERENCE"
                    item["is_far_reference"] = True
                    far_reference_zones.append(item)
                    historical_zones.append(item)
                else:
                    item["is_distant_historical"] = bool(dist_atr > 12.0 and bool(bo_info.get("is_historical_break")))
                    unconfirmed_zones.append(item)
                    if lifecycle in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING"):
                        pending_break_zones.append(item)
        else:
            item["is_active"] = False
            item["zone_relevance"] = "INVALID"
            item["is_far_reference"] = False
            invalidated_zones.append(item)

        bo_state_info = classify_snr_breakout_state(item)
        item["breakout_state_info"] = bo_state_info
        item["breakout_state"] = bo_state_info["state"]

        enriched_all.append(item)

    # Urutkan kandidat aktif berdasarkan ranking_priority tertinggi, lalu deduplikasi overlap lintas TF
    active_sup_candidates.sort(key=lambda x: (-x["ranking_priority"], x["distance_points"], -x.get("score", 0.0), str(x.get("zone_id") or "")))
    active_res_candidates.sort(key=lambda x: (-x["ranking_priority"], x["distance_points"], -x.get("score", 0.0), str(x.get("zone_id") or "")))

    dedup_sup = _deduplicate_overlapping_ranked_zones(active_sup_candidates)
    dedup_res = _deduplicate_overlapping_ranked_zones(active_res_candidates)

    # Untuk penyajian S1, S2: prioritaskan zona berkualitas Grade A/B (>= 70) terlebih dahulu berdasarkan jarak terdekat,
    # sehingga zona Grade C baru di dekat harga tidak menggeser zona utama H1/H4 Grade A/B menjadi S2/R2.
    def _rank_slot_key(x):
        return (
            0 if _is_qualified_ab_zone(x) else 1,
            x["distance_points"],
            -x["ranking_priority"],
            str(x.get("zone_id") or ""),
        )

    top_supports = sorted(dedup_sup[:6], key=_rank_slot_key)
    top_resistances = sorted(dedup_res[:6], key=_rank_slot_key)

    for idx_s, sz in enumerate(top_supports):
        sz["rank_label"] = f"S{idx_s + 1}"
    for idx_r, rz in enumerate(top_resistances):
        rz["rank_label"] = f"R{idx_r + 1}"

    overlaps = []
    for support in top_supports:
        for resistance in top_resistances:
            low = max(safe_float(support.get("low")), safe_float(resistance.get("low")))
            high = min(safe_float(support.get("high")), safe_float(resistance.get("high")))
            if low <= high:
                overlaps.append({
                    "support_id": support.get("zone_id"),
                    "resistance_id": resistance.get("zone_id"),
                    "support_label": support["rank_label"],
                    "resistance_label": resistance["rank_label"],
                    "support_timeframe": support.get("timeframe"),
                    "resistance_timeframe": resistance.get("timeframe"),
                    "low": low,
                    "high": high,
                    "entry_conflict": False,
                })

    broken_zones.sort(key=lambda x: (x["distance_points"], -x.get("score", 0.0)))
    unconfirmed_zones.sort(key=lambda x: (x["distance_points"], -x.get("score", 0.0)))
    pending_break_zones.sort(key=lambda x: (x["distance_points"], -x.get("score", 0.0)))
    reclaimed_zones.sort(key=lambda x: (x["distance_points"], -x.get("score", 0.0)))
    far_reference_zones.sort(key=lambda x: (x["distance_points"], -x.get("score", 0.0)))
    historical_zones.sort(key=lambda x: (x["distance_points"], -x.get("score", 0.0)))
    major_zones.sort(key=lambda x: (-x.get("score", 0.0), x["distance_points"]))

    for k_cat in by_category:
        by_category[k_cat].sort(key=lambda x: (-x["ranking_priority"], -x.get("score", 0.0)))

    return {
        "all_zones": enriched_all,
        "active_supports": top_supports,
        "active_resistances": top_resistances,
        "overlaps": overlaps,
        "major_zones": major_zones,
        "far_reference_zones": far_reference_zones,
        "historical_zones": historical_zones,
        "broken_zones": broken_zones,
        "invalidated_zones": invalidated_zones,
        "unconfirmed_zones": unconfirmed_zones,
        "pending_break_zones": pending_break_zones,
        "reclaimed_zones": reclaimed_zones,
        "by_category": by_category,
    }

