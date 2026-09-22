"""Parses a free-form instruction ("빨간 큐브를 상자에 넣어줘" / "put the red
cube in the box") into a structured {"object": ..., "destination": ...}
target, via the Claude API -- the LLM task-planning layer of the LLM +
open-vocabulary hybrid pipeline (see isaac/hybrid_pick_place_demo.py),
modeled on how Pusan National University's RoboCup@Home-winning team used
LLM-based task planning to decompose free-form commands for their EGPSR
mission.

Deliberately takes no dependency on isaac/object_configs.py or anything
Isaac-Sim-adjacent -- this module only knows "text in, structured JSON out,"
so it's reusable unchanged from the ROS2 deployment side later (vla_bridge)
as well as from this Isaac Sim demo script. Callers pass in the vocabulary
of valid object descriptions explicitly (dependency injection) rather than
this module importing it from elsewhere.

Requires the `anthropic` package (`pip install anthropic`) and an
`ANTHROPIC_API_KEY` environment variable.
"""
import json
import os
import re

MODEL = "claude-sonnet-5"  # ADJUST to whatever Claude model you have API access to

# Matches a ```-fenced block anywhere in the text (optionally tagged, e.g.
# ```json), not just at the very start -- see _extract_json_text.
_FENCE_RE = re.compile(r"```(?:\w*)\s*\n?(.*?)```", re.DOTALL)


class CommandParseError(RuntimeError):
    """Raised when the LLM's response isn't valid JSON, or names an object/
    destination outside the given vocabulary -- callers should treat this as
    "couldn't understand the command," not retry blindly."""


def _extract_json_text(raw_text):
    """Best-effort recovery of a JSON object from an LLM response that may
    not be pure JSON, despite the prompt asking for exactly that.

    2026-09-23: an earlier version of this only stripped a ``` fence when it
    was the very FIRST thing in the response (raw_text.startswith('```')) --
    the single most common real deviation, a leading sentence before the
    fence ("Sure! Here's the JSON:\\n```json\\n{...}"), was not a fence at
    the start of the string at all and fell straight through to the
    json.loads failure path. Tries, in order: (1) the whole response as-is
    (the common case, and the fastest path when the model complies exactly);
    (2) a ```-fenced block anywhere in the response, tagged or not; (3) the
    first balanced {...} object anywhere in the response, respecting quoted
    strings and escapes so a brace inside a string value can't desync the
    match. Returns the extracted text, or the original raw_text unchanged if
    none of the above found anything more specific -- either way, the
    caller's own json.loads is still what actually validates it; this never
    silently invents or fixes JSON, only locates it."""
    stripped = raw_text.strip()
    try:
        json.loads(stripped)
        return stripped  # already valid JSON, nothing to extract
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(raw_text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass  # fall through to the brace scan below

    start = raw_text.find("{")
    if start == -1:
        return stripped
    depth, in_string, escape = 0, False, False
    for i in range(start, len(raw_text)):
        ch = raw_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start:i + 1]
    return stripped  # never balanced -- let json.loads report the real error


def _build_prompt(instruction, object_vocabulary, destination_vocabulary):
    objects_str = ", ".join(object_vocabulary)
    destinations_str = ", ".join(destination_vocabulary)
    return (
        "You control a robot arm that can pick up one object and place it "
        "at one destination. Given the user's instruction, output ONLY a "
        "JSON object (no other text) with exactly two keys:\n"
        f'  "object": one of [{objects_str}]\n'
        f'  "destination": one of [{destinations_str}]\n\n'
        "If the instruction doesn't clearly specify an object from that "
        "exact list, pick the closest match. Never invent an object or "
        "destination outside the given lists.\n\n"
        f'Instruction: "{instruction}"'
    )


def parse_command(instruction, object_vocabulary, destination_vocabulary=("target_zone",)):
    """object_vocabulary: list of description strings matching
    object_configs.all_descriptions() (e.g. ["a red cube", "a blue cube", ...]).
    destination_vocabulary: list of valid destination names -- just
    "target_zone" for the current single-destination scene, but plural so
    this doesn't need to change if the scene later gains more than one.

    Returns {"object": <one of object_vocabulary>, "destination": <one of
    destination_vocabulary>}. Raises CommandParseError on anything that
    doesn't cleanly parse -- deliberately not silently guessing, since a
    wrong grasp target executed on a real robot is worse than stopping and
    asking again."""
    import anthropic  # lazy import: only needed when this function is actually called

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise CommandParseError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(instruction, object_vocabulary, destination_vocabulary)

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    # Guard content[0] being a non-text block explicitly: no tools are
    # passed to this call, so it should always be text, but an unguarded
    # response.content[0].text would raise a bare AttributeError instead of
    # the diagnosable error this module's own docstring promises ("not
    # silently guessing... stopping and asking again" implies a caller can
    # actually tell what went wrong).
    if not response.content or not hasattr(response.content[0], "text"):
        raise CommandParseError(
            f"LLM response's first content block was not text: {response.content!r}")
    raw_text = response.content[0].text.strip()
    json_text = _extract_json_text(raw_text)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise CommandParseError(f"LLM did not return valid JSON: {raw_text!r}") from exc

    if parsed.get("object") not in object_vocabulary:
        raise CommandParseError(f"LLM named an object outside the vocabulary: {parsed!r}")
    if parsed.get("destination") not in destination_vocabulary:
        raise CommandParseError(f"LLM named a destination outside the vocabulary: {parsed!r}")

    return parsed
