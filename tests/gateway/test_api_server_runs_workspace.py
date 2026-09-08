"""Request-scoped workspace binding and toolset narrowing for /v1/runs."""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, adapter  # noqa: F401


class TestWorkspaceBinding:
    @pytest.mark.asyncio
    async def test_workspace_key_must_be_lowercase_identifier(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "session_id": "ws-case-1", "workspace_key": "../evil"},
            )
            assert resp.status == 422
            body = await resp.json()
            assert body["error"]["code"] == "invalid_workspace_key"

    @pytest.mark.asyncio
    async def test_session_workspace_conflict_returns_409(self, adapter, tmp_path):
        from gateway.platforms import api_server_runs

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                with patch.object(api_server_runs, "WORKSPACE_ROOT", tmp_path):
                    first = await cli.post(
                        "/v1/runs",
                        json={
                            "input": "hello",
                            "session_id": "ws-case-2",
                            "workspace_key": "case_a",
                        },
                    )
                    assert first.status == 202
                    second = await cli.post(
                        "/v1/runs",
                        json={
                            "input": "hello",
                            "session_id": "ws-case-2",
                            "workspace_key": "case_b",
                        },
                    )
                    assert second.status == 409
                    body = await second.json()
                    assert body["error"]["code"] == "session_workspace_conflict"

    @pytest.mark.asyncio
    async def test_same_workspace_key_rebinds_after_restart(self, adapter, tmp_path):
        from gateway.platforms import api_server_runs

        api_server_runs._SESSION_WORKSPACE_BINDINGS.pop("ws-case-3", None)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                with patch.object(api_server_runs, "WORKSPACE_ROOT", tmp_path):
                    for _ in range(2):
                        resp = await cli.post(
                            "/v1/runs",
                            json={
                                "input": "hello",
                                "session_id": "ws-case-3",
                                "workspace_key": "case_same",
                            },
                        )
                        assert resp.status == 202
        api_server_runs._SESSION_WORKSPACE_BINDINGS.pop("ws-case-3", None)

    @pytest.mark.asyncio
    async def test_enabled_toolsets_must_be_a_list(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "enabled_toolsets": "not-a-list"},
            )
            assert resp.status == 422

    @pytest.mark.asyncio
    async def test_enabled_toolsets_narrowing_is_forwarded(self, adapter, tmp_path):
        from gateway.platforms import api_server_runs

        api_server_runs._SESSION_WORKSPACE_BINDINGS.pop("ws-case-4", None)
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                with patch.object(api_server_runs, "WORKSPACE_ROOT", tmp_path):
                    resp = await cli.post(
                        "/v1/runs",
                        json={
                            "input": "hello",
                            "session_id": "ws-case-4",
                            "workspace_key": "case_ts",
                            "enabled_toolsets": ["hermes-api-server"],
                        },
                    )
                    assert resp.status == 202
                    assert mock_create.call_args.kwargs.get(
                        "run_enabled_toolsets"
                    ) == ["hermes-api-server"]
        api_server_runs._SESSION_WORKSPACE_BINDINGS.pop("ws-case-4", None)
