"""
MCP Tools — Model Context Protocol tool definitions
Exposes job tracker capabilities as MCP-compatible tools for agent use.
"""

from typing import Any
import json
from datetime import datetime


# ── MCP Tool Schemas ──────────────────────────────────────────────────────────
# These follow the MCP tool definition format so they can be registered
# with any MCP-compatible host (Claude Desktop, OpenAI, etc.)

MCP_TOOLS = [
    {
        "name": "log_job",
        "description": "Log a new job application to the tracker database",
        "inputSchema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name"},
                "title": {"type": "string", "description": "Job title"},
                "url": {"type": "string", "description": "Job listing URL"},
                "match_score": {"type": "integer", "description": "Resume match score 0-100"},
                "salary": {"type": "string", "description": "Salary range if known"},
                "location": {"type": "string", "description": "Job location"},
                "notes": {"type": "string", "description": "Any additional notes"},
            },
            "required": ["company", "title", "url"],
        },
    },
    {
        "name": "update_status",
        "description": "Update the status of an existing job application",
        "inputSchema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name to update"},
                "status": {
                    "type": "string",
                    "enum": ["To Apply", "Applied", "Phone Screen", "Interview", "Offer", "Rejected", "Withdrawn"],
                    "description": "New application status",
                },
                "notes": {"type": "string", "description": "Notes about the status update"},
            },
            "required": ["company", "status"],
        },
    },
    {
        "name": "get_summary",
        "description": "Get a summary of all tracked job applications",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_reminders",
        "description": "Get follow-up reminders for applications that need action",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "search_jobs",
        "description": "Search for new job listings matching given criteria",
        "inputSchema": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Job role to search for"},
                "location": {"type": "string", "description": "Location preference"},
                "tech_stack": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required technologies",
                },
            },
            "required": ["role"],
        },
    },
]


# ── MCP Tool Executor ─────────────────────────────────────────────────────────

def execute_mcp_tool(tool_name: str, arguments: dict) -> Any:
    """
    Execute an MCP tool by name with given arguments.
    This is the dispatcher that routes tool calls to actual implementations.

    Args:
        tool_name: Name of the MCP tool to execute
        arguments: Tool input arguments

    Returns:
        Tool execution result
    """
    from memory.job_store import (
        log_job_to_db,
        update_job_status,
        get_jobs_summary,
        get_pending_followups,
    )
    from tools.search_tool import search_jobs as _search_jobs

    handlers = {
        "log_job": lambda args: log_job_to_db(**args),
        "update_status": lambda args: update_job_status(
            company=args["company"],
            new_status=args["status"],
            notes=args.get("notes", ""),
        ),
        "get_summary": lambda args: get_jobs_summary(),
        "get_reminders": lambda args: get_pending_followups(),
        "search_jobs": lambda args: _search_jobs(
            query=f"{args['role']} {' '.join(args.get('tech_stack', []))}",
            location=args.get("location", "Chennai Remote India"),
        ),
    }

    handler = handlers.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = handler(arguments)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_mcp_tool_schemas() -> list[dict]:
    """Return all MCP tool schemas for registration."""
    return MCP_TOOLS


def format_tool_result(result: Any) -> str:
    """Format tool result as a string for agent consumption."""
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2, default=str)
    return str(result)


# ── Standalone Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich import print_json

    console = Console()
    console.print("[bold blue]🔧 MCP Tool Schemas[/bold blue]")

    for tool in MCP_TOOLS:
        console.print(f"\n[green]{tool['name']}[/green]: {tool['description']}")

    console.print(f"\n[bold]Total tools registered: {len(MCP_TOOLS)}[/bold]")
