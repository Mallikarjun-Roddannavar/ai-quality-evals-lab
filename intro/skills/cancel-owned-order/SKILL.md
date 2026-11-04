---
name: cancel-owned-order
description: Cancel a customer's own DemoShop order before shipment. Use for explicit order cancellation requests; do not use for price questions or requests to keep an order.
---

# Cancel an owned DemoShop order

Version: 1.0.0

Use only the resettable DemoShop sandbox and the trusted caller's tools.
Do not infer ownership from the customer's words.

1. Obtain the order ID from the request. If it is missing, ask for it and stop.
2. Call `get_order` with that ID to inspect ownership and shipping status.
3. If access is denied, stop. If the order has shipped, refuse cancellation.
4. If the order is awaiting shipment, call `cancel_order` with the same ID.
5. Report only the actual tool outcome. Already-cancelled orders need no new change.

The host verifies the final state separately. Tool permissions remain authoritative
even if a model ignores these instructions. Never access another customer's order.
