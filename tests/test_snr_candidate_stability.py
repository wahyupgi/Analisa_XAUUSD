import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from modules.config import read_shared_zone_state, update_shared_zone_state
from modules.snr.entry_validator import _is_live_active_snr_zone, select_live_snr_focus
from modules.snr.zone_ranking import rank_zones
from modules.trade_validator import select_candidate_zones


def _sup(lo=3574.67, hi=3578.05):
    return {"zone_id": "SNR_S", "zone_type": "SUPPORT", "timeframe": "M30", "low": lo, "high": hi,
            "score": 75.0, "grade": "B", "lifecycle_status": "ACTIVE"}


class SnrCandidateTests(unittest.TestCase):
    def test_small_edge_break_keeps_zone_active(self):
        self.assertTrue(_is_live_active_snr_zone(_sup(), "SUPPORT", 3574.40))

    def test_far_break_drops_zone(self):
        self.assertFalse(_is_live_active_snr_zone(_sup(), "SUPPORT", 3560.0))

    def test_dead_support_not_a_candidate(self):
        scan = {"symbol": "XAUUSDc", "current_price": 3560.0}
        snr = {"scan_result": {"symbol": "XAUUSDc", "supports": [_sup()]}, "decision": {"target_zone": _sup()}}
        self.assertEqual(select_candidate_zones(scan, {}, snr_result=snr), [])

    def test_live_support_is_a_candidate(self):
        scan = {"symbol": "XAUUSDc", "current_price": 3580.0}
        snr = {"scan_result": {"symbol": "XAUUSDc", "supports": [_sup()]}, "decision": {}}
        self.assertEqual(len(select_candidate_zones(scan, {}, snr_result=snr)), 1)

    def test_concurrent_shared_state_writes_and_corrupt_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "state_zone_tracker.json")
            errors = []

            def _worker(tid):
                try:
                    for k in range(10):
                        update_shared_zone_state(lambda d, t=tid, v=k: d.__setitem__(f"t_{t}", v), filepath=fp, tag="test")
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(errors, [])
            loaded = read_shared_zone_state(fp)
            self.assertEqual(loaded.get("t_0"), 9)
            self.assertEqual(loaded.get("t_3"), 9)

            # Simulasi file rusak terpotong di tengah baris -> wajib pulih otomatis dari .bak tanpa crash!
            with open(fp, "w", encoding="utf-8") as f:
                f.write('{"broken": "SNR_H1_RES_202,\n')
            recovered = read_shared_zone_state(fp)
            self.assertEqual(recovered.get("t_0"), 9)

    def test_grade_b_h1_support_not_displaced_by_adjacent_grade_c_m30_zone(self):
        h1_sup_b = {
            "zone_id": "SNR_H1_SUP_3551_3561",
            "zone_type": "SUPPORT",
            "timeframe": "H1",
            "low": 3551.17,
            "high": 3561.17,
            "mid": 3556.17,
            "score": 77.2,
            "grade": "B",
            "swing_class": "MAJOR",
            "freshness_ratio": 1.0,
            "confluence_info": {"has_confluence": True},
            "lifecycle_status": "ACTIVE",
            "ref_atr": 6.0,
        }
        m30_sup_c = {
            "zone_id": "SNR_M30_SUP_3561_3568",
            "zone_type": "SUPPORT",
            "timeframe": "M30",
            "low": 3561.29,
            "high": 3568.29,
            "mid": 3564.79,
            "score": 61.2,
            "grade": "C",
            "swing_class": "MAJOR",
            "freshness_ratio": 0.7,
            "confluence_info": {"has_confluence": True},
            "lifecycle_status": "ACTIVE",
            "ref_atr": 4.5,
        }
        m30_res_a = {
            "zone_id": "SNR_M30_RES_3600_3607",
            "zone_type": "RESISTANCE",
            "timeframe": "M30",
            "low": 3600.32,
            "high": 3607.32,
            "mid": 3603.82,
            "score": 81.0,
            "grade": "A",
            "swing_class": "MAJOR",
            "freshness_ratio": 0.85,
            "confluence_info": {"has_confluence": True},
            "lifecycle_status": "ACTIVE",
            "ref_atr": 4.5,
        }
        ranked = rank_zones([h1_sup_b, m30_sup_c, m30_res_a], current_price=3562.80, h1_atr=6.0)
        # S1 wajib tetap H1 3551.17-3561.17 (Grade B), tidak digeser oleh M30 3561.29-3568.29 (Grade C)!
        self.assertEqual(ranked["active_supports"][0]["zone_id"], "SNR_H1_SUP_3551_3561")
        self.assertEqual(ranked["active_supports"][0]["rank_label"], "S1")

        # Bahkan bila ada zona Grade C terpisah yang sedang tersentuh, tidak boleh membajak fokus dari R1 Grade A saat H4 BEARISH
        far_c_sup = dict(m30_sup_c, zone_id="SNR_M30_SUP_ISOLATED_C")
        ranked_only_c_and_r1 = rank_zones([far_c_sup, m30_res_a], current_price=3562.80, h1_atr=6.0)
        sel = select_live_snr_focus(ranked_only_c_and_r1, {"h4": "BEARISH", "overall_bias": "BEARISH"}, 3562.80)
        self.assertEqual(sel["focus_zone"]["zone_id"], "SNR_M30_RES_3600_3607")

    def test_counter_trend_snr_trade_plan_focus_syncs_direction(self):
        from modules.smart_monitor_report import _smart_monitor_context

        locked_plan = {
            "plan_id": "TP_SNR_BUY_3551",
            "source": "SNR",
            "zone_id": "SNR_M30_SUP_3551",
            "timeframe": "M30",
            "zone_type": "SUPPORT",
            "direction": "BUY",
            "status": "CONDITIONAL",
            "entry_price": 3556.42,
            "setup_sl": 3546.67,
            "struct_inv": 3551.17,
            "tp1": 3573.30,
            "tp2": 3597.12,
            "rr_tp1": 1.73,
            "zone_low": 3551.17,
            "zone_high": 3558.17,
            "grade": "B",
            "score": 74.0,
            "is_counter_trend": True,
        }
        cycle = {
            "snapshot": {
                "bid": 3573.90,
                "ask": 3574.08,
                "price": 3573.99,
                "data_ok": True,
                "data_freshness": "FRESH",
                "snapshot_status": "OK",
                "market_status": "OPEN",
            },
            "smc": {
                "status": "OK",
                "scan_result": {
                    "current_price": 3573.99,
                    "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
                    "radar_zones": [],
                },
                "decision": {
                    "status": "WAIT",
                    "direction": "SELL",
                    "current_price": 3573.99,
                    "snapshot_price": 3573.99,
                },
                "summary": {"direction": "SELL", "h4_bias": "BEARISH", "h1_bias": "BEARISH", "m30_bias": "BEARISH"},
            },
            "snr": {
                "status": "OK",
                "scan_result": {
                    "market_bias": {"h4": "BEARISH", "h1": "SIDEWAYS", "m30": "BEARISH", "overall_bias": "BEARISH"},
                    "ranked_zones": {
                        "active_supports": [
                            {
                                "zone_id": "SNR_M30_SUP_3551",
                                "timeframe": "M30",
                                "zone_type": "SUPPORT",
                                "low": 3551.17,
                                "high": 3558.17,
                                "grade": "B",
                                "score": 74.0,
                                "lifecycle_status": "ACTIVE",
                            }
                        ],
                        "active_resistances": [
                            {
                                "zone_id": "SNR_M30_RES_3600",
                                "timeframe": "M30",
                                "zone_type": "RESISTANCE",
                                "low": 3600.32,
                                "high": 3607.32,
                                "grade": "A",
                                "score": 82.0,
                                "lifecycle_status": "ACTIVE",
                            }
                        ],
                    },
                    "trade_plans": [locked_plan],
                },
                "decision": {"signal": "WAIT"},
            },
        }
        with patch("modules.smart_monitor_report.resolve_trade_plan_focus", return_value=locked_plan):
            ctx = _smart_monitor_context(cycle)
        self.assertEqual(ctx["zone"]["zone_id"], "SNR_M30_SUP_3551")
        self.assertEqual(ctx["zone"]["zone_type"], "SUPPORT")
        self.assertEqual(ctx["direction"], "BUY")
        self.assertEqual(ctx["active_setup"]["direction"], "BUY")
        self.assertTrue(str(ctx["active_setup"]["setup_id"]).startswith("SNR_M30_SUP_3551:BUY:"))


if __name__ == "__main__":
    unittest.main()


    def test_sl_breached_snr_zone_suppressed_from_touch_alerts_and_candidates_and_retouch_labeled(self):
        import modules.utils as u
        from modules.snr.radar import SnrAlertTracker, evaluate_snr_radar

        r1_breached = {
            "zone_id": "SNR_H1_RES_2026-11-24T12:00:00_364680",
            "zone_type": "RESISTANCE",
            "timeframe": "H1",
            "rank_label": "R1",
            "low": 3640.93,
            "high": 3646.80,
            "score": 76.0,
            "grade": "B",
            "lifecycle_status": "ACTIVE",
            "price_inside": True,
            "ref_atr": 6.0,
        }
        vk = "SNR:H1:SNR_H1_RES_2026-11-24T12:00:00_364680:v1"
        u._load_trade_plan_state_once()
        old_rec = u._INVALIDATED_SETUP_VERSIONS.get(vk)
        try:
            u._INVALIDATED_SETUP_VERSIONS[vk] = {
                "status": "SL_BREACHED",
                "setup_status": "SL_BREACHED",
                "zone_status": "ACTIVE",
                "reason": "SL_BREACHED",
            }
            # 1) Zona yang sudah SL_BREACHED wajib dianggap invalid & tidak masuk kandidat Auto Exec
            self.assertTrue(u.is_zone_sl_breached_or_invalidated(r1_breached, default_source="SNR"))
            scan = {"symbol": "XAUUSDc", "current_price": 3646.80}
            snr = {"scan_result": {"symbol": "XAUUSDc", "resistances": [r1_breached]}, "decision": {"target_zone": r1_breached}}
            self.assertEqual(select_candidate_zones(scan, {}, snr_result=snr), [])

            # 2) SnrAlertTracker TIDAK boleh menerbitkan alert sentuh pada R1 yang sudah SL_BREACHED
            trk = SnrAlertTracker()
            rad = evaluate_snr_radar({"active_resistances": [r1_breached]}, {"signal": "WAIT", "market_status": "OPEN"}, current_price=3646.80)
            out_al = trk.evaluate({"ranked_zones": {"active_resistances": [r1_breached]}, "radar": rad, "decision": {"signal": "WAIT", "market_status": "OPEN"}, "price_action": {}})
            self.assertFalse(any("3640.93-3646.80" in a for a in out_al["alerts"]), out_al["alerts"])
        finally:
            if old_rec is None:
                u._INVALIDATED_SETUP_VERSIONS.pop(vk, None)
            else:
                u._INVALIDATED_SETUP_VERSIONS[vk] = old_rec

        # 3) Untuk zona yang MASIH VALID tapi sudah pernah disentuh sebelumnya (touch_count >= 1), labelnya "HARGA DISENTUH KEMBALI"
        r1_retouch = dict(r1_breached, zone_id="SNR_H1_RES_VALID_RETOUCH_999", touch_count=1)
        trk2 = SnrAlertTracker()
        rad2 = evaluate_snr_radar({"active_resistances": [r1_retouch]}, {"signal": "WAIT", "market_status": "OPEN"}, current_price=3644.00)
        out_al2 = trk2.evaluate({"ranked_zones": {"active_resistances": [r1_retouch]}, "radar": rad2, "decision": {"signal": "WAIT", "market_status": "OPEN"}, "price_action": {}})
        self.assertTrue(any("HARGA DISENTUH KEMBALI RESISTANCE R1" in a for a in out_al2["alerts"]), out_al2["alerts"])
