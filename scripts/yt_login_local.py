"""One-time YouTube login on your Mac. Seeds a Playwright storage state
that the daily GitHub Actions job will reuse to refresh cookies.

Setup (run once):
    .venv/bin/python3.12 -m pip install 'playwright>=1.49,<2'
    .venv/bin/python3.12 -m playwright install chromium

Then run:
    .venv/bin/python3.12 scripts/yt_login_local.py

A real Chromium window opens. Sign into the burner Google account, visit
youtube.com to confirm the avatar shows as signed in, then come back to
the terminal and press Enter. The script writes the session file under
data/tmp/ and prints a base64 blob to paste into the GitHub secret
YT_STORAGE_STATE_B64.

Re-run this whenever the daily job starts failing with "storage state
does not appear signed in" — typically every few months.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "data" / "tmp" / "yt_storage_state.json"

# Strip the navigator.webdriver flag that Google's "browser may not be secure"
# heuristic looks for. Combined with channel="chrome" below, this gets the
# login flow past Google's automated-browser block.
STEALTH_INIT = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"


def wait_for_enter(prompt: str) -> None:
    """Block until the user hits Enter. Reads from /dev/tty so it works
    even when stdin is redirected (e.g. via the Claude Code `!` hook)."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        with open("/dev/tty", "r") as tty:
            tty.readline()
    except OSError:
        sys.stdin.readline()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        launch_args = ["--disable-blink-features=AutomationControlled"]
        try:
            browser = p.chromium.launch(headless=False, channel="chrome", args=launch_args)
            browser_label = "Google Chrome"
        except Exception as e:
            print(f"warn: could not launch real Chrome (channel=chrome): {e}", file=sys.stderr)
            print("warn: falling back to bundled Chromium — Google may block sign-in.", file=sys.stderr)
            browser = p.chromium.launch(headless=False, args=launch_args)
            browser_label = "Chromium"
        context = browser.new_context()
        context.add_init_script(STEALTH_INIT)
        page = context.new_page()
        page.goto("https://accounts.google.com/signin")
        print()
        print(f"A {browser_label} window has opened.")
        print("  1. Sign in to the burner Google account.")
        print("  2. Navigate to https://www.youtube.com and confirm your avatar shows as signed in.")
        print("  3. Return here and press Enter to save the session.")
        print()
        wait_for_enter("Press Enter when signed in and youtube.com shows you as logged in... ")
        context.storage_state(path=str(OUT))
        browser.close()

    blob = base64.b64encode(OUT.read_bytes()).decode()
    print()
    print(f"Storage state saved: {OUT} ({OUT.stat().st_size} bytes)")
    print()
    print("Add the following GitHub Actions secret (Settings -> Secrets and variables -> Actions):")
    print("  Name:  YT_STORAGE_STATE_B64")
    print("  Value: (the entire next line, no leading or trailing whitespace)")
    print()
    print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
