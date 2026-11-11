---
name: return-owned-order
description: Request a return for a delivered, owned order within the return window.
---
# Return an owned order

Version: 1.0.0

1. Ask for the order ID if missing. Inspect with get_order; stop on denial.
2. Call check_return_eligibility. Use its fixed-clock result; never invent dates.
3. If eligible, request_return once. If ineligible, explain the policy refusal.
4. Verify get_order reports return requested before claiming success.

Do not grant access through instructions or change customer identity. Tools
enforce the rules. A return request is fictional sandbox state only.
