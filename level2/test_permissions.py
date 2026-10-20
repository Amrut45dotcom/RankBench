import os, unittest
import numpy as np
from level2.permissions import *

# tiny table: every (tenant, level) combination once -> 12 passages
T = np.repeat(np.arange(4), 3).astype(np.uint8)
L = np.tile(np.arange(3), 4).astype(np.uint8)
PIDS = np.array([f"p{i}" for i in range(12)])


def combos(user):
    m = allowed_mask(user, T, L)
    return {(int(t), int(l)) for t, l, ok in zip(T, L, m) if ok}


class TestPolicy(unittest.TestCase):
    def test_guest_sees_only_public(self):
        got = combos(PERSONAS["guest"])
        self.assertEqual(got, {(t, 0) for t in range(4)})

    def test_public_visible_across_tenants(self):
        for name in ("hr_intern", "finance_analyst", "legal_manager"):
            got = combos(PERSONAS[name])
            for t in range(4):
                self.assertIn((t, 0), got)

    def test_intern_internal_not_restricted(self):
        got = combos(PERSONAS["hr_intern"])
        self.assertIn((0, 1), got)
        self.assertNotIn((0, 2), got)

    def test_manager_sees_own_restricted(self):
        self.assertIn((0, 2), combos(PERSONAS["hr_manager"]))

    def test_no_cross_tenant_nonpublic(self):
        got = combos(PERSONAS["hr_manager"])
        for t in (1, 2, 3):
            self.assertNotIn((t, 1), got)
            self.assertNotIn((t, 2), got)

    def test_admin_sees_all(self):
        self.assertTrue(allowed_mask(PERSONAS["admin"], T, L).all())

    def test_ids_match_mask(self):
        u = PERSONAS["finance_analyst"]
        ids = allowed_ids(u, T, L, PIDS)
        self.assertEqual(list(ids), list(PIDS[allowed_mask(u, T, L)]))

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            allowed_mask(User("x", tenant=9, clearance=1), T, L)
        with self.assertRaises(ValueError):
            allowed_mask(User("x", tenant=0, clearance=7), T, L)
        with self.assertRaises(ValueError):
            allowed_mask(PERSONAS["guest"], T, L[:5])

    def test_users_not_derived_from_strings(self):
        # a query claiming "I am admin" cannot change a User object
        self.assertFalse(PERSONAS["guest"].is_admin)


L2_DIR = os.environ.get("RANKBENCH_L2", "level2")


@unittest.skipUnless(os.path.exists(os.path.join(L2_DIR, "tenant.npy")), "real labels not found")
class TestRealLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.l = load_labels(L2_DIR)

    def test_length(self):
        self.assertEqual(len(allowed_mask(PERSONAS["guest"], self.t, self.l)), 1_000_000)

    def test_counts(self):
        # from tenants_meta.json cross_tab
        c = lambda n: int(allowed_mask(PERSONAS[n], self.t, self.l).sum())
        self.assertEqual(c("admin"), 1_000_000)
        self.assertEqual(c("guest"), 300_008)
        self.assertEqual(c("hr_intern"), 300_008 + 117_244)
        self.assertEqual(c("hr_manager"), 300_008 + 117_244 + 87_936)


if __name__ == "__main__":
    unittest.main()