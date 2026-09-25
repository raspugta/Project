"""Front-end integrity: every UI string exists in every language; JS files parse."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from baymax.ontology import SUPPORTED_LANGS, ConsentScope

WEB = Path(__file__).resolve().parents[1] / "web"
node = shutil.which("node")


@pytest.mark.skipif(not node, reason="node not installed")
def test_ui_strings_and_consent_texts_complete():
    script = ("import('" + (WEB / "i18n.js").as_uri() + "').then(m => console.log(JSON.stringify("
              "{ui: Object.fromEntries(Object.entries(m.UI).map(([k, v]) => [k, Object.keys(v)])),"
              " consent: Object.fromEntries(Object.entries(m.CONSENT_TEXT).map(([k, v]) => [k, Object.keys(v)]))})))")
    out = json.loads(subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, check=True).stdout)
    assert set(out["ui"]) == set(SUPPORTED_LANGS)
    ref = set(out["ui"]["en"])
    for lang, keys in out["ui"].items():
        assert set(keys) == ref, (lang, ref ^ set(keys))
    for lang, scopes in out["consent"].items():
        assert set(scopes) == {s.value for s in ConsentScope}, lang


@pytest.mark.skipif(not node, reason="node not installed")
@pytest.mark.parametrize("name", ["app.js", "sensors.js", "voice.js", "i18n.js"])
def test_js_parses(name):
    subprocess.run([node, "--check", "--input-type=module"], input=(WEB / name).read_text(), text=True, check=True)


def test_html_references_existing_assets():
    html = (WEB / "index.html").read_text()
    for asset in ("styles.css", "app.js"):
        assert f"/static/{asset}" in html and (WEB / asset).exists()
