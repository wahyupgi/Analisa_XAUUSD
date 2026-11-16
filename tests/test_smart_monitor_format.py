import unittest
from unittest.mock import patch

from modules.unified_monitor import UnifiedMonitor, format_smart_monitor_detail


def _monitor_inputs():
    zone = {
        "zone_id": "H1_SUP_110",
        "timeframe": "H1",
        "zone_type": "SUPPLY",
        "price_low": 110.0,
        "price_high": 112.0,
        "invalidation": 113.0,
        "grade": "A+",
        "score": 90.6,
        "status": "RADAR",
        "radar_eligibility": "QUALIFIED",
    }
    scan = {
        "snapshot_id": "SMART_TEST",
        "snapshot_price": 120.0,
        "radar_zones": [zone],
        "active_zones": [zone],
        "radar_priority": {
            "primary_directional_zone": zone,
            "radar_qualified": [zone],
        },
        "data_freshness": "FRESH",
    }
    decision = {
        "status": "WAIT",
        "direction": None,
        "setup_type": "TREND_PULLBACK",
        "zone": zone,
        "radar_priority": scan["radar_priority"],
        "early_setup": {
            "best_setup": {
                "status": "READY",
                "setup_type": "TREND_PULLBACK",
                "order_type": "SELL LIMIT",
                "suggested_expiration": "2026-11-16 20:00 WIB (8 jam)",
            }
        },
        "stages": {
            "M30": {"status": "CONFIRMED"},
            "zone_touch": {"status": "FAIL"},
            "liquidity_sweep": {"status": "NOT_CHECKED"},
            "structure_trigger": {"status": "NOT_CHECKED"},
            "displacement": {"status": "NOT_CHECKED"},
            "risk_reward": {"status": "NOT_CHECKED"},
        },
        "reasons": ["Harga belum menyentuh zona SUPPLY"],
    }
    summary = {
        "status": "OK",
        "h4_bias": "BEARISH",
        "h1_bias": "BEARISH",
        "m30_bias": "CONFIRMED",
        "m5_bias": "WAIT",
        "primary_zone_label": "H1 SUPPLY 110.00-112.00",
        "decision_signal": "WAIT",
        "direction": "NONE",
        "zone_touch": "FAIL",
        "liquidity_sweep": "WAIT",
        "choch_bos": "WAIT",
        "displacement": "WAIT",
        "rr_stage": "WAIT",
        "signal_status": "WAITING PULLBACK",
        "main_reason": "Menunggu pullback ke supply",
    }
    snr_scan = {
        "snapshot_id": "SMART_TEST",
        "price": 120.0,
        "market_bias": {"overall_bias": "BEARISH"},
        "ranked_zones": {
            "active_supports": [],
            "active_resistances": [{"low": 111.0, "high": 113.0, "rank_label": "R1"}],
        },
        "radar": {},
        "price_action": {},
        "entry_validation": {},
    }
    snr = {
        "method": "SNR",
        "status": "OK",
        "formatted_section": "SNR TEST DETAIL",
        "scan_result": snr_scan,
        "decision": {"signal": "WAIT", "direction": "NONE", "reason": "WAIT"},
        "alerts": [],
    }
    snapshot = {
        "symbol": "XAUUSD",
        "snapshot_id": "SMART_TEST",
        "snapshot_price": 120.0,
        "bid": 119.9,
        "ask": 120.1,
        "spread": 0.2,
        "snapshot_status": "OK",
        "data_ok": True,
        "data_freshness": "FRESH",
        "connection_status": "CONNECTED",
        "server_time": 1794831600,
        "live": False,
        "frames": {},
    }
    return zone, scan, decision, summary, snr, snapshot


class TestSmartMonitorFormat(unittest.TestCase):
    def setUp(self):
        comparison_patcher = patch(
            "modules.unified_monitor.compare_scanner_snapshots",
            return_value={"market_context": {}, "same_snapshot": True},
        )
        comparison_patcher.start()
        self.addCleanup(comparison_patcher.stop)

    def _monitor(self, counts=None):
        zone, scan, decision, summary, snr, snapshot = _monitor_inputs()
        counts = counts if counts is not None else {"smc": 0, "snr": 0}

        def smc_runner(_snapshot):
            counts["smc"] += 1
            return {"scan_result": scan, "decision": decision, "summary": summary}

        def snr_runner(_snapshot):
            counts["snr"] += 1
            return snr

        return UnifiedMonitor(
            symbol="XAUUSD",
            smc_runner=smc_runner,
            snr_runner=snr_runner,
        ), snapshot, zone

    def test_default_report_is_compact_and_does_not_promote_pending_to_signal(self):
        monitor, snapshot, _zone = self._monitor()
        result = monitor.run_cycle(snapshot=snapshot)
        report = result["compact_report"]

        self.assertIn("XAUUSD SMART MONITOR v2.5", report)
        self.assertLessEqual(len(report.splitlines()), 30)
        self.assertIn("PENDING     : READY (NOT PLACED)", report)
        self.assertIn("CONFIRMED   : WAIT", report)
        self.assertIn("SIGNAL      : WAIT", report)
        self.assertIn("SMC + SNR   : ALIGNED", report)

    def test_tick_only_cycle_reuses_scans_and_alerts_zone_approach_once(self):
        counts = {"smc": 0, "snr": 0}
        monitor, snapshot, _zone = self._monitor(counts=counts)
        monitor.run_cycle(snapshot=snapshot)

        tick_snapshot = dict(snapshot, snapshot_price=114.0, bid=113.9, ask=114.1)
        result = monitor.run_cycle(snapshot=tick_snapshot, tick_only=True)
        self.assertEqual(counts, {"smc": 1, "snr": 1})
        self.assertTrue(any("APPROACHING" in alert for alert in result["alerts"]))

        repeated = monitor.run_cycle(snapshot=tick_snapshot, tick_only=True)
        self.assertEqual(repeated["alerts"], [])
        self.assertFalse(repeated["state_changed"])

    def test_detail_report_includes_lifecycle_freshness_and_blockers(self):
        monitor, snapshot, _zone = self._monitor()
        result = monitor.run_cycle(snapshot=snapshot, detailed=True)
        detail = format_smart_monitor_detail(result)

        self.assertIn("LIFECYCLE DETAIL", detail)
        self.assertIn("VALIDATION", detail)
        self.assertIn("FRESHNESS   : FRESH", detail)
        self.assertIn("BLOCKERS", detail)
        self.assertIn("SNR SCANNER DETAIL", detail)

    def test_invalid_data_forces_wait_and_unavailable_price(self):
        monitor, snapshot, _zone = self._monitor()
        snapshot.update(snapshot_status="DATA_INCOMPLETE", data_ok=False, snapshot_price=None)
        result = monitor.run_cycle(snapshot=snapshot)
        self.assertIn("PRICE       : N/A", result["compact_report"])
        self.assertIn("STATE       : WAIT", result["compact_report"])
        self.assertIn("SIGNAL      : WAIT", result["compact_report"])


if __name__ == "__main__":
    unittest.main()
