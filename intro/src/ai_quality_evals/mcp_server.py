"""Actual local DemoShop MCP server: python -m ai_quality_evals.mcp_server."""

from mcp.server.mcpserver import MCPServer

from ai_quality_evals.lab_systems import Shop

server = MCPServer("DemoShop", log_level="ERROR")


@server.tool()
def get_product(product_id: str) -> dict:
    """Look up a public DemoShop product by its exact catalog identifier."""
    return Shop().get_product(product_id)


if __name__ == "__main__":
    server.run(transport="stdio")
