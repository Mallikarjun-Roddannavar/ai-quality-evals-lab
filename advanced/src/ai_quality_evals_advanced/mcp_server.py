"""Actual stdio MCP server; identity and capabilities are trusted host settings."""

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ai_quality_evals_advanced.domain import ShopStore, ToolGateway

server = MCPServer("AdvancedDemoShop", log_level="ERROR")
store = None
gateway = None


def dispatch(name, arguments):
    try:
        return gateway.call(name, arguments)
    except (PermissionError, ValueError) as exc:
        raise ToolError(type(exc).__name__) from None


@server.tool()
async def get_product(product_id: str) -> dict:
    """Look up a public catalog product by its exact ID."""
    return dispatch("get_product", {"product_id": product_id})


@server.tool()
async def get_order(order_id: str) -> dict:
    """Inspect an order belonging to the trusted customer."""
    return dispatch("get_order", {"order_id": order_id})


@server.tool()
async def cancel_order(order_id: str) -> dict:
    """Cancel an owned order before shipment."""
    return dispatch("cancel_order", {"order_id": order_id})


@server.tool()
async def check_return_eligibility(order_id: str) -> dict:
    """Check the fixed-clock fourteen-day return rule."""
    return dispatch("check_return_eligibility", {"order_id": order_id})


@server.tool()
async def request_return(order_id: str) -> dict:
    """Request an eligible return in the isolated sandbox."""
    return dispatch("request_return", {"order_id": order_id})


def main():
    global store, gateway
    from opentelemetry.context import attach, detach
    from opentelemetry.propagate import extract

    from ai_quality_evals_advanced.telemetry import Telemetry

    run_id = os.environ["DEMOSHOP_RUN_ID"]
    folder = Path(os.environ["DEMOSHOP_DATABASE"]).parent / run_id / "mcp-server"
    folder.mkdir(parents=True, exist_ok=True)
    from ai_quality_evals_advanced.contracts import Settings

    endpoint = os.environ.get("DEMOSHOP_PHOENIX")
    if endpoint:
        endpoint = Settings(phoenix_endpoint=endpoint).phoenix_endpoint
    telemetry = Telemetry(folder, run_id, endpoint or None)
    token = attach(extract({"traceparent": os.environ.get("DEMOSHOP_TRACEPARENT", "")}))
    store = ShopStore(Path(os.environ["DEMOSHOP_DATABASE"]))
    gateway = ToolGateway(
        store.session(run_id, os.environ["DEMOSHOP_ACTOR"]),
        set(os.environ["DEMOSHOP_ALLOWED"].split(",")),
        5,
        telemetry,
    )
    try:
        server.run(transport="stdio")
    finally:
        telemetry.close()
        detach(token)
        store.close()


if __name__ == "__main__":
    main()
