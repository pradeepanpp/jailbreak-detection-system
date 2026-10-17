# tests/test_dashboard.py
import os

import pytest

from tests.conftest import ROOT

pytest.importorskip("streamlit")

from src.jailbreak_detection.dashboard import components  # noqa: E402

EVIL = "<img src=x onerror=alert(1)>"


def test_user_text_is_escaped(monkeypatch):
    rendered = []
    monkeypatch.setattr(components.st, "markdown", lambda html, **kw: rendered.append(html))

    components.render_batch_results([{"decision": "BLOCK", "original_text": EVIL, "risk_score": 0.9,
                                      "attack_category": "prompt_injection", "severity": 5}])
    components.render_history([{"decision": "BLOCK", "text": EVIL, "score": 0.9,
                                "category": "prompt_injection"}])

    html = "".join(rendered)
    assert "<img" not in html
    assert "&lt;img" in html


def test_dashboard_starts():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(os.path.join(ROOT, "src/jailbreak_detection/dashboard/app.py"), default_timeout=30).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["ANALYZE", "BATCH", "HISTORY", "ABOUT"]
