"""MCP connection and direct comparison; no model needed for protocol evidence."""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def server_parameters(database, run_id, actor, allowed, telemetry, carrier):
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            s in k.upper()
            for s in ("API_KEY", "SECRET", "LANGSMITH", "LANGCHAIN", "OTEL_")
        )
    }
    env.update(
        DEMOSHOP_RUN_ID=run_id,
        DEMOSHOP_DATABASE=str(database.resolve()),
        DEMOSHOP_ACTOR=actor,
        DEMOSHOP_ALLOWED=",".join(sorted(allowed)),
        DEMOSHOP_TRACEPARENT=carrier.get("traceparent", ""),
        DEMOSHOP_PHOENIX=telemetry.endpoint or "",
    )
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ai_quality_evals_advanced.mcp_server"],
        env=env,
    )


async def mcp_calls(database, run_id, actor, allowed, calls, telemetry):
    from opentelemetry.propagate import inject

    carrier = {}
    with telemetry.span("mcp.client", "TOOL"):
        inject(carrier)
        parameters = server_parameters(
            database, run_id, actor, allowed, telemetry, carrier
        )
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams) as session:
                initialized = await session.initialize()
                listing = await session.list_tools()
                expected = {
                    "get_product",
                    "get_order",
                    "cancel_order",
                    "check_return_eligibility",
                    "request_return",
                }
                if {t.name for t in listing.tools} != expected:
                    raise ValueError("MCP advertised tools changed")
                schemas = {t.name: t.input_schema for t in listing.tools}
                for name, schema in schemas.items():
                    parameter = "product_id" if name == "get_product" else "order_id"
                    if schema.get("required") != [parameter]:
                        raise ValueError("MCP input contract changed")
                results = []
                for name, arguments in calls:
                    with telemetry.span(
                        "mcp.call." + name,
                        "TOOL",
                        **{"input.value": json.dumps(arguments)},
                    ) as span:
                        reply = await session.call_tool(name, arguments)
                        row = {
                            "name": name,
                            "arguments": arguments,
                            "result": reply.structured_content
                            or (
                                json.loads(reply.content[0].text)
                                if not reply.is_error and reply.content
                                else None
                            ),
                            "is_error": reply.is_error,
                        }
                        span.set_attribute("output.value", json.dumps(row))
                        results.append(row)
                return {
                    "protocol": initialized.protocol_version,
                    "schemas": schemas,
                    "results": results,
                    "transport": "real_stdio",
                }


def run_mcp(*args):
    async def bounded():
        return await asyncio.wait_for(mcp_calls(*args), timeout=30)

    return asyncio.run(bounded())
