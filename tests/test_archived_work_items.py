"""Contract tests for the public archived work-item API tools."""

import asyncio
import json

import pytest
from fastmcp import FastMCP
from plane import PlaneClient
from plane.errors.errors import HttpError

from plane_mcp.tools import work_items


class StubResponse:
    """Small requests.Response substitute used by the Plane SDK."""

    def __init__(self, status_code: int, payload=None) -> None:
        self.status_code = status_code
        self.reason = {
            204: "No Content",
            400: "Bad Request",
            403: "Forbidden",
            404: "Not Found",
        }.get(status_code, "OK")
        self._payload = payload
        self.content = b"" if payload is None else json.dumps(payload).encode()
        self.headers = {} if payload is None else {"content-type": "application/json"}

    def json(self):
        return self._payload


def _tool_fn(monkeypatch, client: PlaneClient, tool_name: str):
    monkeypatch.setattr(work_items, "get_plane_client_context", lambda: (client, "hermes"))
    mcp = FastMCP("archived-work-items-test")
    work_items.register_work_item_tools(mcp)
    return asyncio.run(mcp.get_tool(tool_name)).fn


def _page_payload():
    return {
        "results": [
            {
                "id": "work-item-1",
                "name": "Archived item",
                "completed_at": "2026-08-20T10:00:00Z",
                "archived_at": "2026-08-21T10:00:00Z",
            }
        ],
        "total_count": 41,
        "count": 1,
        "next_cursor": "25:1:0",
        "prev_cursor": "",
        "next_page_results": True,
        "prev_page_results": False,
        "total_pages": 2,
        "total_results": 41,
    }


def test_list_archived_work_items_uses_canonical_authenticated_route_and_query(monkeypatch):
    client = PlaneClient(base_url="https://plane.example", api_key="api-secret")
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return StubResponse(200, _page_payload())

    monkeypatch.setattr(client.work_items.session, "get", get)
    tool = _tool_fn(monkeypatch, client, "list_archived_work_items")

    result = tool(
        project_id="project-1",
        pql="   ",
        order_by="-archived_at",
        per_page=25,
        cursor="25:0:0",
        expand="assignees,labels,state,type",
        fields="id,name,completed_at,archived_at",
        filters={
            "priority": "urgent",
            "state_group__in": ["completed", "cancelled"],
        },
    )

    assert calls == [
        (
            "https://plane.example/api/v1/workspaces/hermes/projects/project-1/archived-work-items/",
            {
                "headers": {"Content-Type": "application/json", "X-Api-Key": "api-secret"},
                "params": {
                    "expand": "assignees,labels,state,type",
                    "fields": "id,name,completed_at,archived_at",
                    "order_by": "-archived_at",
                    "cursor": "25:0:0",
                    "per_page": 25,
                    "filters": '{"priority":"urgent","state_group__in":["completed","cancelled"]}',
                },
                "timeout": 30,
            },
        )
    ]
    assert result["results"][0]["archived_at"] == "2026-08-21T10:00:00Z"
    assert result["results"][0]["completed_at"] == "2026-08-20T10:00:00Z"
    assert result["total_count"] == 41
    assert result["count"] == 1
    assert result["next_cursor"] == "25:1:0"
    assert result["prev_cursor"] == ""
    assert result["next_page_results"] is True
    assert result["prev_page_results"] is False
    assert "supported" not in result


def test_list_archived_work_items_rejects_nonempty_pql_before_client_lookup(monkeypatch):
    context_calls = []

    def context_trap():
        context_calls.append(True)
        raise AssertionError("PQL must fail before creating a Plane client")

    monkeypatch.setattr(work_items, "get_plane_client_context", context_trap)
    mcp = FastMCP("archived-work-items-pql-test")
    work_items.register_work_item_tools(mcp)
    tool = asyncio.run(mcp.get_tool("list_archived_work_items")).fn

    result = tool(project_id="project-1", pql='priority = "urgent"')

    assert context_calls == []
    assert result == {
        "error": "PQL is unavailable for archived work-item listing on Plane CE v1.3.1.",
        "unsupported_capability": "archived_work_item_pql",
        "failed_pql": 'priority = "urgent"',
        "hint": "Use the filters parameter with Plane's supported JSON filter fields and operators.",
    }
    assert "pql_reference" not in result
    assert "retry" not in result["hint"].lower()


def test_list_archived_work_items_schema_exposes_filters_and_pql_limit():
    mcp = FastMCP("archived-work-items-schema-test")
    work_items.register_work_item_tools(mcp)
    tool = asyncio.run(mcp.get_tool("list_archived_work_items"))

    properties = tool.parameters["properties"]
    assert properties["filters"]["anyOf"][0] == {
        "additionalProperties": True,
        "type": "object",
    }
    assert "JSON" in properties["filters"]["description"]
    assert "unavailable" in properties["pql"]["description"]
    assert properties["per_page"]["anyOf"][0]["minimum"] == 1
    assert properties["per_page"]["anyOf"][0]["maximum"] == 100
    assert "archived_at" in properties["order_by"]["description"]
    assert "assignees,labels,state,type" in properties["expand"]["description"]
    assert "assignees,labels,state,type" in tool.description


def test_archive_mutation_schemas_document_permissions_and_204_semantics():
    mcp = FastMCP("archive-mutation-schema-test")
    work_items.register_work_item_tools(mcp)

    for tool_name in ("archive_work_item", "unarchive_work_item"):
        tool = asyncio.run(mcp.get_tool(tool_name))
        assert "member or admin" in tool.description
        assert "HTTP 204" in tool.description
        assert "idempotent" in tool.description
        assert "completed_at" in tool.description
        assert "UUID" in tool.parameters["properties"]["project_id"]["description"]
        assert "UUID" in tool.parameters["properties"]["work_item_id"]["description"]


@pytest.mark.parametrize(
    ("tool_name", "method", "url_suffix", "auth_kwargs", "expected_headers", "expected_json"),
    [
        (
            "archive_work_item",
            "post",
            "/archive/",
            {"api_key": "api-secret"},
            {"Content-Type": "application/json", "X-Api-Key": "api-secret"},
            {},
        ),
        (
            "unarchive_work_item",
            "delete",
            "/unarchive/",
            {"access_token": "oauth-secret"},
            {"Content-Type": "application/json", "Authorization": "Bearer oauth-secret"},
            None,
        ),
    ],
)
def test_archive_mutations_use_canonical_routes_and_accept_idempotent_204(
    monkeypatch,
    tool_name,
    method,
    url_suffix,
    auth_kwargs,
    expected_headers,
    expected_json,
):
    client = PlaneClient(base_url="https://plane.example", **auth_kwargs)
    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return StubResponse(204)

    monkeypatch.setattr(client.work_items.session, method, request)
    tool = _tool_fn(monkeypatch, client, tool_name)

    assert tool(project_id="project-1", work_item_id="item-1") is None
    assert tool(project_id="project-1", work_item_id="item-1") is None

    expected_url = f"https://plane.example/api/v1/workspaces/hermes/projects/project-1/work-items/item-1{url_suffix}"
    assert calls == [
        (
            expected_url,
            {"headers": expected_headers, "json": expected_json, "timeout": 30},
        ),
        (
            expected_url,
            {"headers": expected_headers, "json": expected_json, "timeout": 30},
        ),
    ]


@pytest.mark.parametrize("status_code", [400, 403, 404])
@pytest.mark.parametrize(
    ("tool_name", "method"),
    [
        ("list_archived_work_items", "get"),
        ("archive_work_item", "post"),
        ("unarchive_work_item", "delete"),
    ],
)
def test_archive_tools_surface_plane_400_403_404(monkeypatch, tool_name, method, status_code):
    client = PlaneClient(base_url="https://plane.example", api_key="api-secret")

    def request(*args, **kwargs):
        return StubResponse(status_code, {"detail": f"archive error {status_code}"})

    monkeypatch.setattr(client.work_items.session, method, request)
    tool = _tool_fn(monkeypatch, client, tool_name)
    arguments = {"project_id": "project-1"}
    if tool_name != "list_archived_work_items":
        arguments["work_item_id"] = "item-1"

    with pytest.raises(HttpError) as exc_info:
        tool(**arguments)

    assert exc_info.value.status_code == status_code
    assert exc_info.value.response == {"detail": f"archive error {status_code}"}
