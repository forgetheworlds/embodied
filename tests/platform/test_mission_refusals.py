"""The mission's refusal reasons must reach the record that is written out.

``_refusals_log`` accumulated every reason the aircraft did nothing — an
admission the supervisor would not accept, a target that could not be grounded, a
depth product that failed — and then dropped them, because nothing read the list.
Live-19 is the worked case: a frontier resolved to a vantage 5.63 m away,
publication then stopped, and the mission could report only ``no_active_goal``,
because the reason was on a list no artifact carried. Every earlier mission on
this phase lost the same evidence the same way.

These tests hold the two properties that fix it: each distinct refusal appears in
the mission log in the order it happened, and a long list still says how many
were left out rather than truncating in silence.
"""

from __future__ import annotations

from embodied.platform.mission_runtime import MissionResult, MissionRuntime


class _LogHolder:
    """The two attributes ``_surface_refusals`` actually uses.

    The method is called unbound against this stub so the rule can be tested
    without constructing a runtime, whose construction reads the whole platform
    configuration and would start nothing anyway.
    """

    def __init__(self, refusals: list[str]) -> None:
        self._refusals_log = refusals
        self.result = MissionResult(flew=False, termination_reason="mission_completed")


def test_each_distinct_refusal_reaches_the_mission_log_in_order() -> None:
    """A repeated refusal is one reason; the order is the order it happened."""
    holder = _LogHolder(
        [
            "admission refused explore ['frontier:12:2:1']: no_certificate",
            "unsupported_space: the crossing has no supported cells",
            "admission refused explore ['frontier:12:2:1']: no_certificate",
        ]
    )
    MissionRuntime._surface_refusals(holder)
    assert holder.result.log == [
        "refused: admission refused explore ['frontier:12:2:1']: no_certificate",
        "refused: unsupported_space: the crossing has no supported cells",
    ]


def test_a_long_refusal_list_says_what_it_left_out() -> None:
    """Truncation must be reported, never silent."""
    holder = _LogHolder([f"reason-{n}" for n in range(45)])
    MissionRuntime._surface_refusals(holder)
    assert holder.result.log[0] == "refused: reason-0"
    assert holder.result.log[39] == "refused: reason-39"
    assert holder.result.log[40] == "refused: ... and 5 further distinct refusals"
    assert len(holder.result.log) == 41


def test_no_refusals_leaves_the_log_untouched() -> None:
    """A clean mission adds nothing to its own record."""
    holder = _LogHolder([])
    MissionRuntime._surface_refusals(holder)
    assert holder.result.log == []
