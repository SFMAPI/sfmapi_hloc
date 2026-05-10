param(
    [string]$HlocRoot = ".\third_party\hloc",
    [int]$Port = 8000
)

uv run sfmapi-hloc-api --hloc-root $HlocRoot --port $Port --mcp local
