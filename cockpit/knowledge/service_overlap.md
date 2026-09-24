---
doc_id: PB-OVL-01
title: Service overlap with another provider
version: 1.0
signals: service_overlap_other_provider, combo_dup_overlap, combo_contact_overlap
---
## Verify the overlap
Compare service dates, time windows, service codes and care settings for both providers. A case-level overlap flag does not show whether the services were mutually exclusive.

## Legitimate overlap
Some care combinations are expected to overlap, for example adult day care during the day and a home health aide in the evening, or a transition week between two providers. Check the care plan before concluding the services conflict.

## Strong combination
Duplicate billing together with overlap from another provider is the policy strong indicator in this ruleset. Pull service-line records from both providers first.

## Overlap with shared contact details
When the member also shares contact details with a provider, check whether the overlapping provider and the member are linked (same address, phone or family relationship). This pattern can indicate a coordinated arrangement even when no service is billed twice.
