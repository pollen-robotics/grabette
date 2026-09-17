"""Does the prompt do anything?

The camera gets all the attention in policy debugging, but on these
checkpoints language plus state carries roughly 0.7 of the prefix attention
mass -- about twice the camera's share -- and nothing here had ever intervened
on it.

The question is sharp for GRABETTE specifically, because at least one
checkpoint was trained with a prompt that does not describe its task: the
sugar model's training string is "pick up the sugar cup" for a task that
grasps a cube and puts it in a mug. If swapping that for an unrelated string
changes nothing, the policy is effectively single-task and the prompt is
decorative -- which turns the near-collision between that string and pick3's
"pick up the cup" from an annoyance into a real hazard for any future
multi-task merge.

Method is the same intervention as the view ablation, so the numbers are
directly comparable: substitute the prompt, re-run with the SAME noise, and
measure the RMS change in the commanded chunk's translation in millimetres.

Swapping the string is preferred over masking the language tokens. Masking
would change how many tokens the prefix holds and therefore every position
id, so the measurement would confound "different words" with "different
sequence layout". A substituted string is tokenised and padded exactly like
the real one.
"""

from dataclasses import replace
from typing import Iterable

from .metrics import translation_delta, translation_magnitude_mm
from .records import FrameObservation, PromptAblation, PromptSensitivity


def prompt_sensitivity(
    adapter,
    obs: FrameObservation,
    prompts: Iterable[str],
    *,
    noise=None,
) -> PromptSensitivity:
    """Chunk change under each alternative prompt, in millimetres.

    `prompts` are substituted for the frame's own task string one at a time.
    Include the real prompt among them as a control: it must come back at
    exactly 0.0 mm, which is what proves the sweep is measuring the prompt
    rather than sampler noise.

    Every pass shares one noise tensor with the baseline, the same guarantee
    the view ablation and the occlusion sweep rely on.
    """
    noise = adapter.draw_noise() if noise is None else noise
    baseline = adapter.run(obs, noise, capture=False)

    variants = []
    for prompt in prompts:
        swapped = replace(obs, task=prompt)
        result = adapter.run(swapped, noise, capture=False)
        delta = translation_delta(baseline.chunk, result.chunk)
        variants.append(PromptAblation(
            prompt=prompt,
            delta_mm=delta.delta_mm,
            per_axis_mm=delta.per_axis_mm,
        ))

    baseline_mm = translation_magnitude_mm(baseline.chunk)

    return PromptSensitivity(
        baseline_prompt=obs.task,
        baseline_mm=baseline_mm,
        variants=tuple(variants),
    )
