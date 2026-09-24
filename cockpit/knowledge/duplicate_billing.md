---
doc_id: PB-DUP-01
title: Duplicate service billing
version: 1.0
signals: duplicate_service_billed, combo_dup_overlap, combo_contact_dup
---
## Verify the indicator
Compare member, provider, service date, service code and units on each billed line. Check whether one line is a correction, reversal or resubmission of the other. Confirm the payment record before treating a duplicate-service indicator as a duplicate payment.

## What it does not prove
A duplicate flag is a source-system indicator, not a verified event. Keying errors, split billing for long visits and resubmissions after a denial produce the same flag.

## Escalation checkpoint
Escalate only when the duplicate lines were both paid and no correction exists, or when the pattern repeats across several claims for the same provider. Document the evidence and the explanations you ruled out.
