"""_extract_json_text is the part of llm_command_parser.py testable without
a live ANTHROPIC_API_KEY -- everything else in that module is a thin
wrapper around one Claude API call. Each case here is a real deviation
this project found an actual LLM response take, not a hypothetical one:
2026-09-23's adversarial probe of the original startswith("```") fix
caught "Sure! Here is the JSON:\n```json\n{...}```" and "Here you go:
{...}" specifically failing before this file's own fix.
"""
import json

import pytest

from llm_command_parser import _extract_json_text

GOOD = {"object": "a red cube", "destination": "target_zone"}
GOOD_JSON = json.dumps(GOOD)


@pytest.mark.parametrize("label,raw", [
    ("pure JSON, no wrapping", GOOD_JSON),
    ("tagged fence", f"```json\n{GOOD_JSON}\n```"),
    ("untagged fence", f"```\n{GOOD_JSON}\n```"),
    ("fence with trailing prose", f"```json\n{GOOD_JSON}\n```\nLet me know if you need anything else!"),
    ("leading sentence before a tagged fence", f"Sure! Here is the JSON:\n```json\n{GOOD_JSON}\n```"),
    ("leading sentence, no fence at all", f"Here you go: {GOOD_JSON}"),
    ("single-line fence, no newlines", f"```{GOOD_JSON}```"),
])
def test_a_real_llm_response_shape_is_recovered(label, raw):
    extracted = _extract_json_text(raw)
    assert json.loads(extracted) == GOOD, label


def test_a_response_with_no_json_at_all_still_fails_to_parse():
    """The extraction is best-effort, not magic -- it must not invent
    structure that was never there, only locate what is."""
    raw = "I cannot determine which object you mean."
    with pytest.raises(json.JSONDecodeError):
        json.loads(_extract_json_text(raw))


def test_a_brace_inside_a_quoted_string_does_not_desync_the_scan():
    raw = 'Sure: {"object": "a {weird} cube", "destination": "target_zone"}'
    extracted = _extract_json_text(raw)
    assert json.loads(extracted) == {"object": "a {weird} cube", "destination": "target_zone"}
