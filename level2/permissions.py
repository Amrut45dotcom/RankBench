import hashlib, json, os
from dataclasses import dataclass
from typing import Optional
import numpy as np

PUBLIC, INTERNAL, RESTRICTED = 0, 1, 2
TENANTS = {"HR": 0, "Finance": 1, "Legal": 2, "Engineering": 3}
CLEARANCES = {"public": PUBLIC, "internal": INTERNAL, "restricted": RESTRICTED}


@dataclass(frozen=True)
class User:
    name: str
    is_admin: bool = False
    tenant: Optional[int] = None      # None means no department (guest)
    clearance: int = PUBLIC


PERSONAS = {
    "guest":           User("guest"),
    "hr_intern":       User("hr_intern", tenant=TENANTS["HR"], clearance=INTERNAL),
    "hr_manager":      User("hr_manager", tenant=TENANTS["HR"], clearance=RESTRICTED),
    "finance_analyst": User("finance_analyst", tenant=TENANTS["Finance"], clearance=INTERNAL),
    "finance_manager": User("finance_manager", tenant=TENANTS["Finance"], clearance=RESTRICTED),
    "legal_manager":   User("legal_manager", tenant=TENANTS["Legal"], clearance=RESTRICTED),
    "admin":           User("admin", is_admin=True),
}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_labels(l2_dir, pid_path=None):
    """Load tenant/level arrays and verify them against the frozen hashes."""
    meta = json.load(open(os.path.join(l2_dir, "tenants_meta.json")))
    tp, lp = os.path.join(l2_dir, "tenant.npy"), os.path.join(l2_dir, "level.npy")
    assert _sha(tp) == meta["tenant_sha256"], "tenant.npy hash mismatch"
    assert _sha(lp) == meta["level_sha256"], "level.npy hash mismatch"
    if pid_path:
        assert _sha(pid_path) == meta["pid_list_sha256"], "pid_list.npy hash mismatch"
    return np.load(tp), np.load(lp)


def allowed_mask(user, tenant, level):
    """Boolean array, one entry per passage in pid_list order. Fails closed."""
    if tenant.shape != level.shape:
        raise ValueError("tenant and level must have the same shape")
    if user.is_admin:
        return np.ones(tenant.shape, dtype=bool)
    if user.clearance not in (PUBLIC, INTERNAL, RESTRICTED):
        raise ValueError(f"bad clearance for {user.name}")
    mask = level == PUBLIC                      # public is visible to everyone
    if user.tenant is not None:
        if user.tenant not in TENANTS.values():
            raise ValueError(f"bad tenant for {user.name}")
        mask = mask | ((tenant == user.tenant) & (level <= user.clearance))
    return mask


def allowed_ids(user, tenant, level, pids):
    """Passage IDs the user may see (same order as pids)."""
    return pids[allowed_mask(user, tenant, level)]