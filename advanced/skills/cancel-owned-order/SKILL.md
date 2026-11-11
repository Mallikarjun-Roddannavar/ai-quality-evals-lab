---
name: cancel-owned-order
description: Cancel a customer's identified order before shipment.
---
# Cancel an owned order

Version: 1.0.0

1. Ask for the order ID if missing; never guess it.
2. Inspect the order with get_order. If access is denied, stop.
3. If awaiting shipment, call cancel_order once. Otherwise explain the refusal.
4. Verify the resulting order status with get_order before claiming cancellation.

Host authorization and policy override all instructions. Never ask tools to
impersonate another customer. Do not execute scripts or arbitrary commands.
