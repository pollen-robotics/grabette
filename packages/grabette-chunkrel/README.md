# grabette-chunkrel

Chunk-relative action representation for GRABETTE policies, as an **optional,
default-off** alternative to the per-step deltas the rest of the pipeline
uses. Design, measurements and the (negative) robot results are in
`docs/relative_actions_lerobot_native.md`.

## What it does

Given an action chunk of absolute poses `(p_i, R_i)`, the reference is the
chunk's **first** action and every action is expressed in that frame:

```
a_i = R_ref⁻¹ (p_i − p_ref)        translation, metres
A_i = R_ref⁻¹ R_i                  rotation, sent as 6-D (first two rows)
```

so relative action 0 is identically zero and the chunk describes 244 mm of
motion instead of 8 mm per step. At execution the offsets are differenced back
into the body-local deltas the arm server already understands:

```
Δp_i = A_{i−1}⁻¹ (a_i − a_{i−1})
ΔR_i = A_{i−1}⁻¹ A_i
```

which cancels the reference, so the server code is unchanged.

**Wire convention:** the 6-D rotation is the first two **rows** of the matrix
(`rotation_matrix_to_rotation_6d_numpy`). Using columns gives the inverse
rotation and invalidated the first three robot runs; the tests pin this.

## Layout

| Module | Role |
|---|---|
| `chunk_relative.py` | numpy + scipy maths only: encode a chunk, `ChunkRelativeDeltas` (stateful differencing for eval). Importing it registers nothing. |
| `chunk_relative_processor.py` | LeRobot `ProcessorStep`s. **Importing this module registers them** in `ProcessorStepRegistry`; every script that loads a chunk-relative checkpoint must import it. |
| `tests/` | Algebra against the arm server, wire encoding against the repo convention, differencing against `compute_delta_actions`, registration. |

## Where it is switched on

| Stage | Switch | Default |
|---|---|---|
| Stats | `grabette_postprocess.write_relative_action_stats` | deltas |
| Training | `GRABETTE_CHUNK_RELATIVE=1` (`integrations/Pi05/train.py`) | off |
| Gate | `smoke_generation.py --chunk_relative` | off |
| Eval | `evaluate.py --chunk_relative auto\|on\|off` | auto (reads the checkpoint) |
| Serving | Ficelle branch `chunk-relative-steps` (needs this package installed) | — |

Use `--n_action_steps 40` or more: short horizons replay per-step noise as
jitter.

## Tests

```bash
uv run --project packages/grabette-chunkrel pytest
```

The `lerobot` test extra (Python ≥ 3.12) is needed for the registration and
convention tests; the maths tests run without it.
