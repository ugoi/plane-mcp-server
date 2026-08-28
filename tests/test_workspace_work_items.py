"""Compatibility tests for workspace-wide work item listing."""

import asyncio
from types import SimpleNamespace

import pytest
from fastmcp import FastMCP
from plane.errors.errors import HttpError

from plane_mcp.tools import work_items


def _page(results, *, total_count=None):
    count = len(results)
    total = count if total_count is None else total_count
    return SimpleNamespace(
        results=results,
        total_count=total,
        count=count,
        next_cursor="",
        prev_cursor="",
        next_page_results=False,
        prev_page_results=False,
    )


def _tool_fn(client):
    mcp = FastMCP("workspace-work-items-test")
    work_items.register_work_item_tools(mcp)
    tool = asyncio.run(mcp.get_tool("list_workspace_work_items"))
    return tool.fn, client


def test_workspace_list_falls_back_to_only_project_on_404(monkeypatch):
    class FakeWorkItems:
        def __init__(self):
            self.workspace_params = None
            self.project_calls = []

        def list_workspace(self, **kwargs):
            self.workspace_params = kwargs["params"]
            raise HttpError("HTTP 404: Not Found", 404, "Not Found")

        def list(self, **kwargs):
            self.project_calls.append(kwargs)
            return _page([{"id": "item-1", "project": "project-1"}], total_count=212)

    class FakeProjects:
        def __init__(self):
            self.calls = []

        def list(self, **kwargs):
            self.calls.append(kwargs)
            return _page(
                [SimpleNamespace(id="project-1", name="Tasks", identifier="TASK")],
                total_count=1,
            )

    client = SimpleNamespace(work_items=FakeWorkItems(), projects=FakeProjects())
    monkeypatch.setattr(work_items, "get_plane_client_context", lambda: (client, "hermes"))
    tool, _ = _tool_fn(client)

    result = tool(
        pql='priority = "urgent"',
        order_by="-created_at",
        per_page=1,
        cursor="1:0:0",
        expand="state",
        fields="id,name,sequence_id,project",
        external_id="external-1",
        external_source="test",
    )

    assert result == {
        "results": [{"id": "item-1", "project": "project-1"}],
        "total_count": 212,
        "count": 1,
        "next_cursor": "",
        "prev_cursor": "",
        "next_page_results": False,
        "prev_page_results": False,
        "compatibility_fallback": "single_project",
    }
    assert len(client.projects.calls) == 1
    project_params = client.projects.calls[0]["params"]
    assert project_params.per_page == 2
    assert project_params.fields == "id,name,identifier"
    assert len(client.work_items.project_calls) == 1
    project_call = client.work_items.project_calls[0]
    assert project_call["workspace_slug"] == "hermes"
    assert project_call["project_id"] == "project-1"
    assert project_call["params"] is client.work_items.workspace_params
    assert project_call["params"].model_dump(exclude_none=True) == {
        "expand": "state",
        "fields": "id,name,sequence_id,project",
        "external_id": "external-1",
        "external_source": "test",
        "order_by": "-created_at",
        "cursor": "1:0:0",
        "per_page": 1,
        "pql": 'priority = "urgent"',
    }


def test_workspace_list_keeps_cloud_path_when_supported(monkeypatch):
    class FakeWorkItems:
        def list_workspace(self, **kwargs):
            return _page([{"id": "cloud-item"}], total_count=1)

    class UnexpectedProjects:
        def list(self, **kwargs):
            raise AssertionError("project fallback must not run")

    client = SimpleNamespace(work_items=FakeWorkItems(), projects=UnexpectedProjects())
    monkeypatch.setattr(work_items, "get_plane_client_context", lambda: (client, "cloud"))
    tool, _ = _tool_fn(client)

    result = tool(per_page=1)

    assert result == {
        "results": [{"id": "cloud-item"}],
        "total_count": 1,
        "count": 1,
        "next_cursor": "",
        "prev_cursor": "",
        "next_page_results": False,
        "prev_page_results": False,
    }


def test_workspace_list_reports_multi_project_contract_gap(monkeypatch):
    class FakeWorkItems:
        def list_workspace(self, **kwargs):
            raise HttpError("HTTP 404: Not Found", 404, "Not Found")

        def list(self, **kwargs):
            raise AssertionError("a multi-project fallback must not fake global pagination")

    projects = [
        SimpleNamespace(id="project-1", name="One", identifier="ONE"),
        SimpleNamespace(id="project-2", name="Two", identifier="TWO"),
    ]
    client = SimpleNamespace(
        work_items=FakeWorkItems(),
        projects=SimpleNamespace(list=lambda **kwargs: _page(projects, total_count=2)),
    )
    monkeypatch.setattr(work_items, "get_plane_client_context", lambda: (client, "community"))
    tool, _ = _tool_fn(client)

    result = tool()

    assert result["unsupported_capability"] == "workspace_work_item_list"
    assert result["project_count"] == 2
    assert [project["id"] for project in result["projects"]] == ["project-1", "project-2"]
    assert "list_work_items" in result["hint"]


def test_workspace_list_returns_empty_page_when_no_projects(monkeypatch):
    class FakeWorkItems:
        def list_workspace(self, **kwargs):
            raise HttpError("HTTP 404: Not Found", 404, "Not Found")

        def list(self, **kwargs):
            raise AssertionError("there is no project to query")

    client = SimpleNamespace(
        work_items=FakeWorkItems(),
        projects=SimpleNamespace(list=lambda **kwargs: _page([], total_count=0)),
    )
    monkeypatch.setattr(work_items, "get_plane_client_context", lambda: (client, "empty"))
    tool, _ = _tool_fn(client)

    result = tool()

    assert result == {
        "results": [],
        "total_count": 0,
        "count": 0,
        "next_cursor": "",
        "prev_cursor": "",
        "next_page_results": False,
        "prev_page_results": False,
        "compatibility_fallback": "no_projects",
    }


def test_workspace_list_preserves_pql_error_on_project_fallback(monkeypatch):
    class FakeWorkItems:
        def __init__(self):
            self.project_params = None

        def list_workspace(self, **kwargs):
            raise HttpError("HTTP 404: Not Found", 404, "Not Found")

        def list(self, **kwargs):
            self.project_params = kwargs["params"]
            raise HttpError("HTTP 400: Bad Request", 400, {"pql": "Invalid PQL"})

    project = SimpleNamespace(id="project-1", name="Tasks", identifier="TASK")
    client = SimpleNamespace(
        work_items=FakeWorkItems(),
        projects=SimpleNamespace(list=lambda **kwargs: _page([project], total_count=1)),
    )
    monkeypatch.setattr(work_items, "get_plane_client_context", lambda: (client, "community"))
    tool, _ = _tool_fn(client)

    result = tool(pql='priority = "not-a-priority"')

    assert result["error"] == "Invalid PQL"
    assert result["failed_pql"] == 'priority = "not-a-priority"'
    assert "pql_reference" in result
    assert client.work_items.project_params.pql == 'priority = "not-a-priority"'


def test_workspace_list_propagates_non_404(monkeypatch):
    class FakeWorkItems:
        def list_workspace(self, **kwargs):
            raise HttpError("HTTP 403: Forbidden", 403, "Forbidden")

    class UnexpectedProjects:
        def list(self, **kwargs):
            raise AssertionError("project fallback must not run")

    client = SimpleNamespace(work_items=FakeWorkItems(), projects=UnexpectedProjects())
    monkeypatch.setattr(work_items, "get_plane_client_context", lambda: (client, "forbidden"))
    tool, _ = _tool_fn(client)

    with pytest.raises(HttpError) as exc_info:
        tool()

    assert exc_info.value.status_code == 403
