---
doc_id: PB-FP-01
title: Common false-positive explanations
version: 1.0
signals: duplicate_service_billed, service_overlap_other_provider, shared_contact_with_provider, weekly_visit_frequency, prior_claims_last_12mo
---
## Alternative explanations
Corrected or reversed claims, legitimate shared contact details, overlapping but distinct services, high-need members and small peer groups all produce misleading alerts. Check each explanation against source records; the case-level file cannot establish any of them.

## Recording a false positive
When rejecting a finding, record which explanation applied and what record confirmed it. These reasons become the feedback corpus used to improve rules and future retrieval.
