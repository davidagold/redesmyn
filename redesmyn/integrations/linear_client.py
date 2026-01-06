from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, cast

import httpx

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

logger = logging.getLogger("redesmyn.integrations.linear")
_GRAPHQL_OPERATION_RE = re.compile(
    r"^\\s*(query|mutation)\\s+(?P<name>[A-Za-z0-9_]+)\\b"
)

_shared_graphql_client: httpx.AsyncClient | None = None


def _graphql_http_client() -> httpx.AsyncClient:
    # Linear GraphQL calls can happen in bursts (sync, automation). Reusing a
    # shared client keeps connections warm and avoids paying TLS handshake costs
    # on every request.
    global _shared_graphql_client
    if _shared_graphql_client is None or _shared_graphql_client.is_closed:
        _shared_graphql_client = httpx.AsyncClient(timeout=30.0)
    return _shared_graphql_client


class LinearApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        operation: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.operation = operation
        self.request_id = request_id


@dataclass(frozen=True, slots=True)
class LinearClient:
    access_token: str

    async def graphql(
        self, query: str, variables: dict[str, object] | None = None
    ) -> dict[str, Any]:
        operation_match = _GRAPHQL_OPERATION_RE.search(query)
        operation_name = (
            operation_match.group("name") if operation_match is not None else None
        )
        vars_payload = variables or {}

        safe_vars: dict[str, object] = {}
        for key, value in vars_payload.items():
            if isinstance(value, str) and len(value) > 256:
                safe_vars[key] = f"{value[:256]}…"
            elif isinstance(value, list) and len(value) > 20:
                preview: list[object] = []
                for idx, item in enumerate(value):
                    if idx >= 20:
                        break
                    preview.append(item)
                preview.append("…")
                safe_vars[key] = preview
            else:
                safe_vars[key] = value

        def _truncate_obj(obj: object, *, limit: int = 4000) -> str:
            try:
                text = repr(obj)
            except Exception:
                text = str(obj)
            if len(text) <= limit:
                return text
            return f"{text[:limit]}…"

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        client = _graphql_http_client()
        resp = await client.post(
            LINEAR_GRAPHQL_URL,
            json={"query": query, "variables": vars_payload},
            headers=headers,
        )

        payload: object | None
        raw_text: str | None = None
        try:
            payload = resp.json()
        except Exception:
            payload = None
            raw_text = resp.text

        request_id = resp.headers.get("x-request-id")
        if resp.status_code >= 400:
            details = payload if payload is not None else (raw_text or "<no body>")
            logger.error(
                "Linear GraphQL HTTP error: status=%s request_id=%s op=%s vars=%s response=%s",
                resp.status_code,
                request_id,
                operation_name,
                safe_vars,
                _truncate_obj(details),
            )

            if isinstance(payload, dict) and isinstance(payload.get("errors"), list):
                first = payload["errors"][0] if payload["errors"] else None
                if isinstance(first, dict):
                    msg = first.get("message")
                    extensions = first.get("extensions")
                    code = (
                        extensions.get("code") if isinstance(extensions, dict) else None
                    )
                    status_code = (
                        extensions.get("statusCode")
                        if isinstance(extensions, dict)
                        else None
                    )
                    if not isinstance(status_code, int):
                        status_code = resp.status_code
                    raise LinearApiError(
                        str(msg) if msg is not None else "Linear GraphQL error",
                        code=str(code) if isinstance(code, str) else None,
                        status_code=status_code,
                        operation=operation_name,
                        request_id=request_id,
                    )

            raise LinearApiError(
                f"Linear GraphQL HTTP error (status={resp.status_code})",
                status_code=resp.status_code,
                operation=operation_name,
                request_id=request_id,
            )

        if not isinstance(payload, dict):
            raw_text = raw_text if raw_text is not None else resp.text
            logger.error(
                "Linear GraphQL response was not JSON: status=%s request_id=%s op=%s vars=%s response=%s",
                resp.status_code,
                request_id,
                operation_name,
                safe_vars,
                _truncate_obj(raw_text),
            )
            raise ValueError("Linear GraphQL response missing data")

        if "errors" in payload:
            logger.error(
                "Linear GraphQL error: request_id=%s op=%s vars=%s errors=%s",
                request_id,
                operation_name,
                safe_vars,
                _truncate_obj(payload.get("errors")),
            )
            first = None
            errors = payload.get("errors")
            if isinstance(errors, list) and errors:
                first = errors[0]
            if isinstance(first, dict):
                msg = first.get("message")
                extensions = first.get("extensions")
                code = extensions.get("code") if isinstance(extensions, dict) else None
                status_code = (
                    extensions.get("statusCode")
                    if isinstance(extensions, dict)
                    else None
                )
                if not isinstance(status_code, int):
                    status_code = None
                raise LinearApiError(
                    str(msg) if msg is not None else "Linear GraphQL error",
                    code=str(code) if isinstance(code, str) else None,
                    status_code=status_code,
                    operation=operation_name,
                    request_id=request_id,
                )

            raise LinearApiError(
                "Linear GraphQL error",
                operation=operation_name,
                request_id=request_id,
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("Linear GraphQL response missing data")
        return cast(dict[str, Any], data)
