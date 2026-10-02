"""The depth check states where its declared pose came from, and is refused when it cannot.

The measurement itself is the P01-C pipeline, covered by its own tests. What is new
here is the provenance of the declared pose, which decides whether a reader may treat
the declared depth as independent of the estimator: the compat evidence's pose is the
simulator scene's declared static start, while a capture whose pose came from the run's
own published estimate is not, and labelling that as scene truth is the one error this
project treats as disqualifying.
"""

from __future__ import annotations

import pytest

from embodied.contracts import records as R
from embodied.perception.camera import DIAGNOSTIC, declared_pose_provenance

COMPAT_WORLD = "scenarios/compat/worlds/compat_stereo.wbt"


def test_the_default_remains_the_compat_static_start_label():
    provenance = declared_pose_provenance({"world": COMPAT_WORLD})
    assert provenance.label == DIAGNOSTIC
    assert "declared static start" in provenance.detail
    assert COMPAT_WORLD in provenance.detail


def test_a_measured_pose_is_labelled_sensor_derived_and_carries_its_detail():
    provenance = declared_pose_provenance(
        {
            "world": "scenarios/first_indoor/world.wbt",
            "pose_source": "measured_published_pose",
            "pose_source_detail": "the run's own published estimator pose at each sim time",
        }
    )
    assert provenance.label == "SENSOR_DERIVED"
    assert "estimator pose" in provenance.detail


def test_a_measured_pose_that_cannot_state_itself_is_refused():
    with pytest.raises(R.RecordError):
        declared_pose_provenance(
            {
                "world": "scenarios/first_indoor/world.wbt",
                "pose_source": "measured_published_pose",
                "pose_source_detail": "   ",
            }
        )


def test_an_unknown_pose_source_is_not_silently_accepted_as_scene_truth():
    # An unrecognised value must not fall through to the scene-truth label by accident:
    # it is treated as the default only when no measured source is claimed at all.
    provenance = declared_pose_provenance(
        {"world": COMPAT_WORLD, "pose_source": "something_else"}
    )
    assert provenance.label == DIAGNOSTIC
