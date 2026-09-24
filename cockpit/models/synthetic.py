"""DEVELOPMENT ONLY. Documented synthetic, labelled evaluation set (spec 9.4 / 21).
Never used by the running app; see cockpit/models/supervised_model.py.

The 50 supplied cases have no outcome labels, so no real accuracy can be reported on them.
This generator creates a separate labelled set with explicit, documented assumptions so the
rules model and challenger models can be compared on the same ground. Results measure how
well each method recovers *these assumed schemes*, not real-world fraud accuracy.

Assumptions (edit freely; they are the thing to challenge in review):
- Prevalence: 15% suspicious.
- Benign members: signals drawn around the "normal" region of the supplied queue.
- Benign high-need members (12% of benign): high visits, many prior claims, higher amounts,
  but no billing-integrity flags. These are the classic false positives.
- Suspicious schemes:
    duplicate_billing  - duplicate and overlapping services, round-dollar and weekend billing
    collusion          - shared contact details, long distance, high frequency
    subtle_upcoding    - moderately high amounts and round-dollar billing only (hard to catch)
- 3% label noise.
"""
from __future__ import annotations

import numpy as np

CARE_TYPES = ["Adult Day Care", "Assisted Living", "Home Health Aide", "In-Home Care", "Skilled Nursing"]
GENERATOR_VERSION = "synthetic-eval-1.0"


def _beta(rng, mean, k=12):
    return float(np.clip(rng.beta(mean * k, (1 - mean) * k), 0, 1))


def _benign(rng, high_need=False):
    c = {
        "duplicate_service_billed": int(rng.random() < 0.04),
        "service_overlap_other_provider": int(rng.random() < 0.04),
        "shared_contact_with_provider": int(rng.random() < 0.03),
        "recent_policy_change_flag": int(rng.random() < 0.10),
        "weekly_visit_frequency": int(max(1, rng.poisson(4))),
        "member_provider_distance_miles": int(max(1, rng.lognormal(np.log(17), 0.55))),
        "prior_claims_last_12mo": int(rng.poisson(1.7)),
        "weekend_billing_ratio": round(_beta(rng, 0.09), 2),
        "amount_vs_peer_avg_pct": int(rng.normal(0, 11)),
        "round_dollar_billing_ratio": round(_beta(rng, 0.12), 2),
    }
    if high_need:
        c["weekly_visit_frequency"] = int(rng.integers(8, 15))
        c["prior_claims_last_12mo"] = int(rng.integers(5, 11))
        c["amount_vs_peer_avg_pct"] = int(rng.integers(25, 65))
        c["weekend_billing_ratio"] = round(_beta(rng, 0.25), 2)
    return c


def _suspicious(rng, scheme):
    c = _benign(rng)
    if scheme == "duplicate_billing":
        c.update({
            "duplicate_service_billed": int(rng.random() < 0.85),
            "service_overlap_other_provider": int(rng.random() < 0.70),
            "shared_contact_with_provider": int(rng.random() < 0.35),
            "round_dollar_billing_ratio": round(_beta(rng, 0.62), 2),
            "weekend_billing_ratio": round(_beta(rng, 0.40), 2),
            "amount_vs_peer_avg_pct": int(rng.integers(40, 170)),
        })
    elif scheme == "collusion":
        c.update({
            "shared_contact_with_provider": int(rng.random() < 0.90),
            "member_provider_distance_miles": int(rng.integers(60, 250)),
            "weekly_visit_frequency": int(rng.integers(8, 19)),
            "recent_policy_change_flag": int(rng.random() < 0.60),
            "prior_claims_last_12mo": int(rng.integers(3, 12)),
        })
    else:  # subtle_upcoding
        c.update({
            "amount_vs_peer_avg_pct": int(rng.integers(25, 75)),
            "round_dollar_billing_ratio": round(_beta(rng, 0.42), 2),
        })
    return c


def generate(n: int = 3000, prevalence: float = 0.15, noise: float = 0.03, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        suspicious = rng.random() < prevalence
        if suspicious:
            scheme = rng.choice(["duplicate_billing", "collusion", "subtle_upcoding"], p=[0.45, 0.35, 0.20])
            c = _suspicious(rng, scheme)
        else:
            high_need = rng.random() < 0.12
            scheme = "benign_high_need" if high_need else "benign"
            c = _benign(rng, high_need)
        label = int(suspicious)
        if rng.random() < noise:
            label = 1 - label
        c.update({
            "case_id": f"S{i:05d}",
            "claim_number": f"SYN-{i:06d}",
            "claim_date": "2026-01-01",
            "care_type": str(rng.choice(CARE_TYPES)),
            "claim_amount_usd": int(max(300, rng.lognormal(np.log(5000), 0.8))),
            "state": "NA",
            "label": label,
            "scheme": str(scheme),
        })
        rows.append(c)
    return rows
