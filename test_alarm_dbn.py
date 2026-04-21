import unittest

import numpy as np

from alarm_dbn import build_alarm_dbn, load_alarm_bn, unroll_alarm_dbn


class AlarmDBNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_bn = load_alarm_bn()
        cls.dbn = build_alarm_dbn(cls.static_bn, persistence=0.8)

    def test_parser_loads_alarm_structure(self):
        self.assertEqual(len(self.static_bn.node_order), 37)
        self.assertEqual(len(self.static_bn.edges), 46)
        self.assertEqual(self.static_bn.nodes["VentTube"].parents, ("Disconnect", "VentMach"))
        self.assertEqual(self.static_bn.nodes["VentTube"].states, ("Zero", "Low", "Normal", "High"))

    def test_transition_cpds_include_temporal_self_parent(self):
        for name in self.static_bn.node_order:
            transition_cpd = self.dbn.transition_cpds[name]
            self.assertEqual(transition_cpd.parents[0], f"prev_{name}")
            self.assertEqual(transition_cpd.parents[1:], self.static_bn.nodes[name].parents)

    def test_all_cpd_distributions_are_normalized(self):
        for cpd_group in (self.dbn.prior_cpds, self.dbn.transition_cpds):
            for cpd in cpd_group.values():
                for distribution in cpd.table.values():
                    self.assertTrue(np.isclose(distribution.sum(), 1.0))
                    self.assertTrue(np.all(distribution >= 0.0))

    def test_unrolled_counts_match_expected_shape(self):
        unrolled = unroll_alarm_dbn(self.dbn, horizon=3)
        self.assertEqual(len(unrolled.nodes), 111)
        self.assertEqual(len(unrolled.edges), 212)
        self.assertEqual(len(unrolled.cpds), 111)

    def test_unrolled_smoke_build(self):
        unrolled = unroll_alarm_dbn(self.dbn, horizon=3)
        self.assertIn("HR_2", unrolled.nodes)
        self.assertIn(("HR_1", "HR_2"), unrolled.edges)
        self.assertIn("HR_2", unrolled.cpds)


if __name__ == "__main__":
    unittest.main()
