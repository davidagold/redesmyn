from __future__ import annotations

import argparse
import asyncio
import threading
import webbrowser
from dataclasses import dataclass
from datetime import UTC
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from redesmyn.context import get_repo_context
from redesmyn.integrations.linear import (
    LinearApiError,
    LinearClient,
    exchange_code_for_token,
    linear_authorize_url,
    linear_redirect_uri,
    new_oauth_state,
    new_pkce_verifier,
    pkce_code_challenge,
)
from redesmyn.integrations.linear_credentials import default_linear_credential_store
from redesmyn.settings import load_settings


@dataclass(slots=True)
class Callback:
    code: str | None = None
    state: str | None = None
    error: str | None = None


def _parse_scope_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    scope = (qs.get("scope") or [None])[0]
    return scope if isinstance(scope, str) and scope else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Linear OAuth diagnostics: print authorize URL + scope, run auth, print token scopes."
    )
    ap.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Seconds to wait for the OAuth callback.",
    )
    args = ap.parse_args()

    try:
        ctx = get_repo_context()
        repo_root = ctx.repo_root
    except Exception:
        repo_root = Path.cwd()

    settings = load_settings(repo_root=repo_root)
    redirect_uri = linear_redirect_uri(settings)
    state = new_oauth_state()
    code_verifier = new_pkce_verifier()
    code_challenge = pkce_code_challenge(code_verifier)

    authorize_url = linear_authorize_url(
        settings,
        state=state,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
    )

    print(f"repo_root: {repo_root}")
    print(f"redirect_uri: {redirect_uri}")
    print(f"requested_scopes_raw: {settings.linear_scopes!r}")
    print(f"authorize_url_scope_param: {_parse_scope_from_url(authorize_url)!r}")
    print(f"authorize_url: {authorize_url}")
    print("")

    callback = Callback()
    got_callback = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/v1/linear/oauth/callback":
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            callback.code = (qs.get("code") or [None])[0]
            callback.state = (qs.get("state") or [None])[0]
            callback.error = (qs.get("error") or [None])[0]
            got_callback.set()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"<h1>Linear connected</h1><p>You can close this tab and return to the terminal.</p>"
            )

    host = settings.api_host
    port = settings.api_port
    try:
        httpd = HTTPServer((host, port), Handler)
    except OSError as e:
        print(f"error: could not bind OAuth callback listener on {host}:{port} ({e})")
        print(
            "This usually means another process (often `just dev` / the API server) is already listening.\n"
            "Stop whatever is using that port, then rerun this script.\n"
            "Tip: `lsof -nP -iTCP:9234 -sTCP:LISTEN`"
        )
        return 2
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        if not args.no_open:
            if not webbrowser.open(authorize_url):
                print(authorize_url)

        if not got_callback.wait(args.timeout):
            print("error: timed out waiting for OAuth callback")
            return 2

        if callback.error:
            print(f"error: Linear authorization failed: {callback.error}")
            return 2

        if not callback.code or not callback.state:
            print("error: missing code/state in callback")
            return 2
        if callback.state != state:
            print("error: state mismatch in callback")
            return 2

        async def run() -> int:
            try:
                token = await exchange_code_for_token(
                    settings,
                    code=callback.code or "",
                    redirect_uri=redirect_uri,
                    code_verifier=code_verifier,
                )
            except Exception as e:
                print(f"error: token exchange failed: {e}")
                return 2

            print(f"token.scope: {token.scope!r}")
            print(f"token.expires_at: {token.expires_at!r}")
            print("")

            store = default_linear_credential_store()
            creds = store.get()
            if creds is not None:
                print(f"keychain.scope: {creds.scope!r}")
                print(
                    f"keychain.connected_at: {creds.connected_at.astimezone(UTC).isoformat()}"
                )
                print("")

            client = LinearClient(access_token=token.access_token)
            try:
                await client.graphql("query { viewer { id name email } }")
                print("viewer query: ok")
            except LinearApiError as e:
                print(
                    f"viewer query: error: {e} (code={e.code}, status={e.status_code})"
                )
                return 2

            # Write-scope probe (best-effort, side-effect free):
            # attempt to create a label that already exists. Without `write` this
            # should fail with "Invalid scope: `write` required". With `write`,
            # it should fail with a "already exists" style error (or return success=false).
            probe_name = None
            try:
                data = await client.graphql(
                    "query { issueLabels(first: 1) { nodes { name } } }"
                )
                conn = data.get("issueLabels")
                nodes = conn.get("nodes") if isinstance(conn, dict) else None
                first = nodes[0] if isinstance(nodes, list) and nodes else None
                probe_name = (
                    first.get("name")
                    if isinstance(first, dict) and isinstance(first.get("name"), str)
                    else None
                )
            except LinearApiError:
                probe_name = None

            if not probe_name:
                print("write probe: could not find existing label name to probe")
                return 0

            probe_query = (
                "mutation Probe($name: String!) { "
                "issueLabelCreate(input: { name: $name }) { success } "
                "}"
            )
            try:
                data = await client.graphql(probe_query, variables={"name": probe_name})
                payload = data.get("issueLabelCreate")
                success = payload.get("success") if isinstance(payload, dict) else None
                print(
                    f"write probe: mutation returned (likely has write): success={success!r}"
                )
            except LinearApiError as e:
                msg = str(e)
                if "Invalid scope" in msg or (e.code or "").upper() == "FORBIDDEN":
                    print("write probe: forbidden (missing write scope)")
                else:
                    print(f"write probe: got error (likely has write): {e}")

            return 0

        return asyncio.run(run())
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
