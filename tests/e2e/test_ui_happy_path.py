from __future__ import annotations

import os

import pytest
from playwright.sync_api import sync_playwright


@pytest.mark.e2e
@pytest.mark.timeout(120)
def test_ui_happy_path_loads_graph_and_opens_task_details(
    e2e_base_url: str,
    e2e_epic_slug: str,
    e2e_epic_id: int,
) -> None:
    headless = os.environ.get("REDESMYN_E2E_HEADFUL") != "1"
    errors: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(base_url=e2e_base_url)
        page = context.new_page()

        page.on("pageerror", lambda exc: errors.append(str(exc)))

        page.goto(f"/graph/{e2e_epic_slug}", wait_until="domcontentloaded")

        node_cards = page.locator("[data-node-card]")
        node_cards.first.wait_for(state="visible", timeout=30_000)

        node_cards.first.click()
        page.get_by_role("button", name="README").wait_for(
            state="visible", timeout=10_000
        )

        with page.expect_response(
            lambda resp: f"/v1/epics/{e2e_epic_id}/graph" in resp.url
        ) as resp_info:
            page.get_by_role("button", name="Refresh").click()
        resp = resp_info.value
        if resp.status != 200:
            raise AssertionError(
                f"Refresh did not succeed (status={resp.status}, url={resp.url})"
            )

        if errors:
            raise AssertionError(f"page errors: {errors}")

        context.close()
        browser.close()
