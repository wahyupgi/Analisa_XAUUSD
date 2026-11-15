# -*- coding: utf-8 -*-
"""
zone_scoring.py — Evaluasi Multi-Timeframe Confluence & Sistem Scoring Zona Independen (0-100)
untuk SNR Advanced Scanner.

Komponen Penilaian Wajib (Total = 100):
1. Swing / Structural Strength  : 20
2. Historical Price Reaction    : 20
3. Touch Quality                : 15
4. Multi-TF Confluence          : 15
5. Rejection / Breakout Quality : 15
6. Freshness                    : 10
7. Momentum Confirmation        : 5

Grade:
- Grade A : 80.00 - 100.00
- Grade B : 70.00 - 79.99
- Grade C : 60.00 - 69.99
- Grade D : < 60.00
"""
from modules.snr.config import get_snr_config
from modules.snr.data_provider import safe_float


def _zones_overlap_or_close(z1, z2, gap_tol):
    """Mengecek apakah dua zona bertipe sama saling beririsan atau berada dalam toleransi gap."""
    l1, h1 = safe_float(z1.get("low"), 0.0), safe_float(z1.get("high"), 0.0)
    l2, h2 = safe_float(z2.get("low"), 0.0), safe_float(z2.get("high"), 0.0)
    gap = max(l1, l2) - min(h1, h2)
    return gap <= gap_tol


def evaluate_mtf_confluence(zones_by_tf, config=None):
    """
    Mengidentifikasi Multi-Timeframe Confluence antar timeframe struktural (H4, H1, M30)
    tanpa memaksakan penggabungan zona yang terlalu lebar dan tanpa menghitung timeframe
    yang sama berkali-kali sebagai bukti independen.

    Mengembalikan:
    - confluence_groups: list of dict (area confluence lintas TF)
    - zone_confluence_map: dict {zone_id: confluence_info}
    """
    cfg = config or get_snr_config()
    conf_cfg = cfg.get("confluence", {})
    gap_tol_atr = safe_float(conf_cfg.get("confluence_gap_tolerance_atr", 0.22), 0.22)
    max_width_pts = safe_float(conf_cfg.get("max_confluence_zone_width_points", 14.0), 14.0)

    structural_tfs = ("H4", "H1", "M30")
    all_struct_zones = []
    for tf in structural_tfs:
        for z in zones_by_tf.get(tf, []) or []:
            if z.get("lifecycle_status", "ACTIVE") in ("ACTIVE", "BROKEN"):
                if safe_float(z.get("width"), 0.0) <= max_width_pts:
                    all_struct_zones.append(z)

    zone_confluence_map = {}
    confluence_groups = []
    seen_group_signatures = set()

    for anchor in all_struct_zones:
        a_id = anchor.get("zone_id")
        a_tf = str(anchor.get("timeframe", "H1")).upper()
        a_type = str(anchor.get("zone_type", "SUPPORT")).upper()
        a_atr = max(0.10, safe_float(anchor.get("ref_atr"), 2.0))
        gap_tol = a_atr * gap_tol_atr

        # Pilih maksimal 1 zona terbaik yang beririsan dari setiap timeframe lain (mencegah double counting)
        tf_matches = {a_tf: anchor}
        for other_tf in structural_tfs:
            if other_tf == a_tf:
                continue
            candidates_tf = [
                oz for oz in all_struct_zones
                if str(oz.get("timeframe", "")).upper() == other_tf
                and str(oz.get("zone_type", "")).upper() == a_type
                and bool(oz.get("is_active", True)) == bool(anchor.get("is_active", True))
                and _zones_overlap_or_close(anchor, oz, gap_tol)
            ]
            if candidates_tf:
                best_other = min(
                    candidates_tf,
                    key=lambda oz: abs(safe_float(oz.get("mid"), 0.0) - safe_float(anchor.get("mid"), 0.0)),
                )
                tf_matches[other_tf] = best_other

        matched_tfs = [tf for tf in structural_tfs if tf in tf_matches]
        if len(matched_tfs) <= 1:
            zone_confluence_map[a_id] = {
                "has_confluence": False,
                "timeframes": [a_tf],
                "tf_count": 1,
                "confluence_low": anchor.get("low"),
                "confluence_high": anchor.get("high"),
                "score_15": 4.5 if a_tf == "H4" else (3.5 if a_tf == "H1" else 2.5),
                "reason": f"Single-TF level ({a_tf}) tanpa irisan langsung dengan timeframe lain.",
            }
            continue

        matched_zones = [tf_matches[t] for t in matched_tfs]

        # Hitung pita confluence inti (union dari pairwise intersections jika overlap, atau irisan terdekat)
        pairwise_lows = []
        pairwise_highs = []
        n_m = len(matched_zones)
        for i in range(n_m):
            for j in range(i + 1, n_m):
                ov_low = max(safe_float(matched_zones[i]["low"], 0.0), safe_float(matched_zones[j]["low"], 0.0))
                ov_high = min(safe_float(matched_zones[i]["high"], 0.0), safe_float(matched_zones[j]["high"], 0.0))
                if ov_high >= ov_low:
                    pairwise_lows.append(ov_low)
                    pairwise_highs.append(ov_high)

        if pairwise_lows and pairwise_highs:
            conf_low = round(min(pairwise_lows), 2)
            conf_high = round(max(pairwise_highs), 2)
        else:
            mids = [safe_float(z["mid"], 0.0) for z in matched_zones]
            conf_low = round(min(mids), 2)
            conf_high = round(max(mids), 2)

        if (conf_high - conf_low) > max_width_pts:
            # Jangan memaksakan penggabungan zona yang terlalu lebar
            conf_low = anchor.get("low")
            conf_high = anchor.get("high")

        # Hitung skor Multi-TF Confluence (0 - 15)
        if len(matched_tfs) == 3:
            score_15 = 15.0
        elif "H4" in matched_tfs and "H1" in matched_tfs:
            score_15 = 13.0
        elif "H4" in matched_tfs and "M30" in matched_tfs:
            score_15 = 12.0
        else:  # H1 + M30
            score_15 = 10.5

        reason = (
            f"Multi-TF {a_type} Confluence ({'+'.join(matched_tfs)}) "
            f"pada rentang inti {conf_low:.2f} - {conf_high:.2f}."
        )

        conf_info = {
            "has_confluence": True,
            "zone_type": a_type,
            "timeframes": matched_tfs,
            "tf_count": len(matched_tfs),
            "confluence_low": conf_low,
            "confluence_high": conf_high,
            "member_zone_ids": [z.get("zone_id") for z in matched_zones],
            "score_15": round(score_15, 2),
            "reason": reason,
        }
        zone_confluence_map[a_id] = conf_info

        sig = (a_type, tuple(matched_tfs), conf_low, conf_high)
        if sig not in seen_group_signatures:
            seen_group_signatures.add(sig)
            confluence_groups.append(conf_info)

    return {
        "confluence_groups": confluence_groups,
        "zone_confluence_map": zone_confluence_map,
    }


def assign_snr_grade(total_score, config=None):
    """Menentukan Grade A / B / C / D berdasarkan total skor (0-100)."""
    cfg = config or get_snr_config()
    grades = cfg.get("grades", {"A": 80.0, "B": 70.0, "C": 60.0})
    s = safe_float(total_score, 0.0)
    if s >= safe_float(grades.get("A", 80.0), 80.0):
        return "A"
    if s >= safe_float(grades.get("B", 70.0), 70.0):
        return "B"
    if s >= safe_float(grades.get("C", 60.0), 60.0):
        return "C"
    return "D"


def score_zone(
    zone,
    confluence_info=None,
    breakout_info=None,
    retest_info=None,
    rejection_info=None,
    momentum_info=None,
    config=None,
):
    """
    Menghitung skor zona SNR (0-100) secara objektif dan dapat diaudit pada 7 komponen wajib:
    1. Swing / Structural Strength  (maks 20)
    2. Historical Price Reaction    (maks 20)
    3. Touch Quality                (maks 15)
    4. Multi-TF Confluence          (maks 15)
    5. Rejection / Breakout Quality (maks 15)
    6. Freshness                    (maks 10)
    7. Momentum Confirmation        (maks 5)
    """
    cfg = config or get_snr_config()
    tf = str(zone.get("timeframe", "H1")).upper()
    zone_cfg = cfg.get("zone", {}).get(tf, cfg.get("zone", {}).get("H1", {}))
    tf_weight = safe_float(zone_cfg.get("tf_weight", 0.85), 0.85)

    # 1. Swing / Structural Strength (0 - 20)
    sw_strength = safe_float(zone.get("swing_strength"), 5.0)
    sw_class = str(zone.get("swing_class", "MINOR")).upper()
    consol = str(zone.get("consolidation_structure", "STANDARD_PIVOT")).upper()

    base_struct = min(15.0, sw_strength * 1.55)
    class_bonus = 2.5 if sw_class == "MAJOR" else 0.8
    struct_bonus = 1.5 if consol in ("IMPULSIVE_DEPARTURE", "CONSOLIDATION_BASE") else 0.5
    comp_swing = min(20.0, round((base_struct + class_bonus + struct_bonus) * (0.82 + 0.18 * tf_weight), 2))
    reason_swing = (
        f"Swing {sw_class} (strength={sw_strength:.2f}/10, struktur={consol}, TF={tf}) -> {comp_swing:.2f}/20."
    )

    # 2. Historical Price Reaction (0 - 20)
    bounce_atr = max(
        safe_float(zone.get("bounce_strength_atr"), 0.0),
        safe_float((zone.get("touch_analysis") or {}).get("max_bounce_atr"), 0.0),
    )
    hist_cnt = int(zone.get("historical_reaction_count", 0))
    rep_rej = int(zone.get("repeated_rejections", 0))

    # Hindari double counting: fokus pada magnitudo reaksi harga (bounce_atr) dan konsistensi pantulan
    mag_points = min(15.0, bounce_atr * 8.5)
    react_bonus = min(5.0, (min(2, hist_cnt) * 1.6) + (min(2, rep_rej) * 0.9))
    comp_reaction = min(20.0, round(mag_points + react_bonus, 2))
    reason_reaction = (
        f"Kekuatan pantulan historis maks {bounce_atr:.2f} ATR dengan {hist_cnt} reaksi valid -> {comp_reaction:.2f}/20."
    )

    # 3. Touch Quality (0 - 15)
    touch_data = zone.get("touch_analysis") or {}
    touch_status = str(zone.get("touch_status", touch_data.get("classification", "UNTESTED"))).upper()
    comp_touch = min(15.0, max(0.0, safe_float(touch_data.get("touch_quality_score_15"), 11.0)))
    reason_touch = f"{touch_data.get('reason', f'Status sentuhan {touch_status}')} -> {comp_touch:.2f}/15."

    # 4. Multi-TF Confluence (0 - 15)
    if confluence_info is None:
        confluence_info = zone.get("confluence_info") or {}
    comp_confluence = min(
        15.0,
        max(
            0.0,
            safe_float(
                confluence_info.get("score_15"),
                4.5 if tf == "H4" else (3.5 if tf == "H1" else 2.5),
            ),
        ),
    )
    reason_confluence = (
        f"{confluence_info.get('reason', f'Level mandiri {tf}')} -> {comp_confluence:.2f}/15."
    )

    # 5. Rejection / Breakout Quality (0 - 15)
    rej_q = safe_float((rejection_info or {}).get("quality_score_15"), 0.0)
    bo_q = safe_float((breakout_info or {}).get("quality_score_15"), 0.0)
    rt_status = str((retest_info or {}).get("status", "NO_RETEST")).upper()

    if rt_status == "RETEST_CONFIRMED":
        comp_rej_bo = min(15.0, max(13.0, bo_q, rej_q))
        reason_rej_bo = f"Breakout + Retest terkonfirmasi ({retest_info.get('role_reversal')}) -> {comp_rej_bo:.2f}/15."
    elif rej_q > 0 or bo_q > 0:
        comp_rej_bo = min(15.0, max(rej_q, bo_q))
        active_desc = (
            rejection_info.get("status")
            if rej_q >= bo_q and rejection_info
            else (breakout_info.get("status") if breakout_info else "NONE")
        )
        reason_rej_bo = f"Kualitas sinyal aksi harga ({active_desc}) -> {comp_rej_bo:.2f}/15."
    else:
        # Kualitas penolakan struktural saat pembentukan zona jika harga belum menguji ulang saat ini
        comp_rej_bo = min(11.5, round(min(10.0, bounce_atr * 6.0) + (1.5 if sw_class == "MAJOR" else 0.5), 2))
        reason_rej_bo = (
            f"Penolakan struktural pembentuk zona ({bounce_atr:.2f} ATR, belum ada trigger M5 baru) -> {comp_rej_bo:.2f}/15."
        )

    # 6. Freshness (0 - 10)
    fresh_ratio = safe_float(zone.get("freshness_ratio"), 0.75)
    if touch_status == "UNTESTED":
        comp_freshness = round(min(10.0, 8.0 + (2.0 * fresh_ratio)), 2)
        reason_freshness = f"Zona UNTESTED (freshness ratio={fresh_ratio:.2f}) -> {comp_freshness:.2f}/10."
    elif touch_status == "TESTED_1X":
        comp_freshness = round(min(9.0, 6.8 + (2.0 * fresh_ratio)), 2)
        reason_freshness = f"Di-test 1x dan bertahan baik (freshness={fresh_ratio:.2f}) -> {comp_freshness:.2f}/10."
    elif touch_status == "TESTED_2X":
        comp_freshness = round(min(7.8, 5.5 + (2.0 * fresh_ratio)), 2)
        reason_freshness = f"Di-test 2x (freshness moderat={fresh_ratio:.2f}) -> {comp_freshness:.2f}/10."
    elif touch_status == "TESTED_3X_PLUS":
        comp_freshness = round(max(2.0, 3.5 * fresh_ratio), 2)
        reason_freshness = f"Telah diuji >=3x (freshness menurun={fresh_ratio:.2f}) -> {comp_freshness:.2f}/10."
    elif touch_status == "BROKEN":
        comp_freshness = 4.0 if rt_status in ("RETEST_WAITING", "RETEST_IN_PROGRESS", "RETEST_CONFIRMED") else 1.5
        reason_freshness = f"Zona asli telah ditembus ({touch_status}, retest={rt_status}) -> {comp_freshness:.2f}/10."
    else:
        comp_freshness = 0.0
        reason_freshness = "Zona INVALIDATED (freshness=0) -> 0.00/10."

    # 7. Momentum Confirmation (0 - 5)
    if momentum_info and isinstance(momentum_info, dict):
        mom_score = safe_float(momentum_info.get("score_5"), 2.0)
        mom_dir = str(momentum_info.get("direction", "NEUTRAL")).upper()
        z_type = str(zone.get("zone_type", "SUPPORT")).upper()
        # Sesuaikan apakah momentum mendukung pertahanan zona atau breakout
        if (z_type == "SUPPORT" and mom_dir == "BULLISH") or (z_type == "RESISTANCE" and mom_dir == "BEARISH"):
            comp_momentum = min(5.0, round(mom_score, 2))
        elif rt_status == "RETEST_CONFIRMED":
            comp_momentum = min(5.0, round(mom_score, 2))
        else:
            comp_momentum = min(3.8, round(mom_score * 0.80, 2))
        reason_momentum = f"Momentum {mom_dir} ({momentum_info.get('strength', 'N/A')}) -> {comp_momentum:.2f}/5."
    else:
        comp_momentum = 2.5
        reason_momentum = "Momentum netral/standar -> 2.50/5."

    raw_total = (
        comp_swing
        + comp_reaction
        + comp_touch
        + comp_confluence
        + comp_rej_bo
        + comp_freshness
        + comp_momentum
    )

    # Aturan wajib Section N:
    # 1. Zona INVALIDATED tidak boleh mendapat skor tinggi.
    # 2. Jangan memberikan Grade A hanya karena banyak touch (jika TESTED_3X_PLUS dengan tanda absorpsi, batasi < 80).
    absorption_warn = bool(touch_data.get("absorption_warning", False))
    if touch_status == "INVALIDATED" or zone.get("lifecycle_status") == "INVALIDATED":
        total_score = min(35.0, round(raw_total * 0.4, 2))
    elif touch_status == "BROKEN" and rt_status not in ("RETEST_CONFIRMED", "RETEST_IN_PROGRESS", "RETEST_WAITING"):
        total_score = min(58.0, round(raw_total * 0.68, 2))
    elif touch_status == "TESTED_3X_PLUS" and (absorption_warn or not bool(confluence_info.get("has_confluence", False))):
        total_score = min(78.5, round(raw_total, 2))
    else:
        total_score = min(100.0, max(0.0, round(raw_total, 2)))

    grade = assign_snr_grade(total_score, config=cfg)

    breakdown = {
        "swing_structural_strength": {
            "score": comp_swing,
            "max": 20.0,
            "reason": reason_swing,
        },
        "historical_price_reaction": {
            "score": comp_reaction,
            "max": 20.0,
            "reason": reason_reaction,
        },
        "touch_quality": {
            "score": comp_touch,
            "max": 15.0,
            "reason": reason_touch,
        },
        "multi_tf_confluence": {
            "score": comp_confluence,
            "max": 15.0,
            "reason": reason_confluence,
        },
        "rejection_breakout_quality": {
            "score": comp_rej_bo,
            "max": 15.0,
            "reason": reason_rej_bo,
        },
        "freshness": {
            "score": comp_freshness,
            "max": 10.0,
            "reason": reason_freshness,
        },
        "momentum_confirmation": {
            "score": comp_momentum,
            "max": 5.0,
            "reason": reason_momentum,
        },
        "total_score": total_score,
        "grade": grade,
        "note": "Skor zona (0-100) terpisah dari validasi entry; Grade A tidak otomatis berarti sinyal entry aktif.",
    }

    scored_zone = dict(zone)
    scored_zone["score"] = total_score
    scored_zone["score_int"] = int(round(total_score))
    scored_zone["grade"] = grade
    scored_zone["score_breakdown"] = breakdown
    scored_zone["confluence_info"] = confluence_info
    if breakout_info is not None:
        scored_zone["breakout_info"] = breakout_info
    if retest_info is not None:
        scored_zone["retest_info"] = retest_info
    if rejection_info is not None:
        scored_zone["rejection_info"] = rejection_info
    return scored_zone

