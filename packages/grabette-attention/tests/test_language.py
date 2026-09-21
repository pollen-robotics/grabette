"""Prompt sensitivity: swap the language, measure the chunk's movement.

The fake adapter's chunk depends on the prompt string in a known way, so the
sweep must recover that dependence and nothing else. The control matters most:
re-running with the frame's OWN prompt has to come back at exactly zero, or
the measurement is picking up sampler noise instead of the language.
"""
import numpy as np
import pytest

from grabette_attention.adapters.base import RunResult
from grabette_attention.language import prompt_sensitivity
from grabette_attention.records import FrameObservation

REAL = "pick up the sugar cup"


class FakeAdapter:
    """Commands z proportional to the prompt's length, in metres."""

    def __init__(self):
        self.prompts: list[str] = []
        self.noise_draws = 0

    @property
    def camera_keys(self):
        return ("cam0",)

    def draw_noise(self):
        self.noise_draws += 1
        return "fixed-noise"

    def run(self, obs, noise, *, capture, drop_camera=None):
        assert noise == "fixed-noise", "every pass must share one noise tensor"
        self.prompts.append(obs.task)
        chunk = np.zeros((10, 8), np.float32)
        chunk[:, 2] = len(obs.task) * 1e-3
        return RunResult(chunk=chunk)


def observation(task=REAL):
    return FrameObservation(
        episode=0, frame=48, images={"cam0": np.zeros((240, 320, 3), np.uint8)},
        state=np.zeros(2, np.float32), task=task,
    )


def test_the_real_prompt_is_a_zero_control():
    # The single most important check: substituting the prompt the frame
    # already carries must change nothing at all.
    result = prompt_sensitivity(FakeAdapter(), observation(), [REAL])
    assert result.variants[0].delta_mm == 0.0


def test_a_different_prompt_moves_the_chunk():
    result = prompt_sensitivity(
        FakeAdapter(), observation(), [REAL, "pick up the cup"]
    )
    control, other = result.variants
    assert control.delta_mm == 0.0
    # The fake's z scales with prompt length: 21 vs 15 chars = 6 mm.
    assert other.delta_mm == pytest.approx(6.0)


def test_one_result_per_prompt_in_order():
    prompts = [REAL, "", "close the drawer", "pick up the red can"]
    result = prompt_sensitivity(FakeAdapter(), observation(), prompts)
    assert [v.prompt for v in result.variants] == prompts


def test_one_pass_per_prompt_plus_one_baseline():
    adapter = FakeAdapter()
    prompts = ["a", "bb", "ccc"]
    prompt_sensitivity(adapter, observation(), prompts)
    assert adapter.prompts == [REAL, "a", "bb", "ccc"]


def test_the_noise_is_drawn_once():
    adapter = FakeAdapter()
    prompt_sensitivity(adapter, observation(), ["a", "b", "c"])
    assert adapter.noise_draws == 1


def test_a_supplied_noise_tensor_is_reused():
    adapter = FakeAdapter()
    prompt_sensitivity(
        adapter, observation(), ["a"], noise="fixed-noise"
    )
    assert adapter.noise_draws == 0


def test_the_callers_frame_keeps_its_own_prompt():
    obs = observation()
    prompt_sensitivity(FakeAdapter(), obs, ["something else entirely"])
    assert obs.task == REAL


def test_an_empty_prompt_is_a_legitimate_variant():
    # "no instruction at all" is the cleanest test of whether the prompt
    # matters, so it must not be filtered out as falsy.
    result = prompt_sensitivity(FakeAdapter(), observation(), [""])
    assert result.variants[0].prompt == ""
    assert result.variants[0].delta_mm == pytest.approx(21.0)


def test_the_baseline_magnitude_is_reported_for_scale():
    # A 21-character prompt gives 21 mm of z on every step, so the RMS
    # translation magnitude is 21 mm.
    result = prompt_sensitivity(FakeAdapter(), observation(), [REAL])
    assert result.baseline_mm == pytest.approx(21.0)
    assert result.baseline_prompt == REAL


def test_the_per_axis_breakdown_is_carried_through():
    result = prompt_sensitivity(FakeAdapter(), observation(), [""])
    # The fake only moves z, which is slot 2.
    x, y, z = result.variants[0].per_axis_mm
    assert (x, y) == pytest.approx((0.0, 0.0))
    assert z == pytest.approx(21.0)


def test_no_prompts_gives_no_variants_and_still_reports_the_baseline():
    result = prompt_sensitivity(FakeAdapter(), observation(), [])
    assert result.variants == ()
    assert result.baseline_mm == pytest.approx(21.0)
