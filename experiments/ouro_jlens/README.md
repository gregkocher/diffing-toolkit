# Ouro recurrence-resolved Jacobian lens

This experiment fits the reference `jlens.fit` estimator from the personal
`gregkocher/jacobian-lens` fork, then runs the existing toolkit `main.py` diff
mining pipeline. The pair is the false-cake adapter versus original Ouro.
The base-model lens is shared by both arms (`lens_source=base`).

The native final physical block is reused four times. `OuroLensModel` inserts
identity hook boundaries at the output of each invocation, before recurrent
normalization. It calls the native text model unchanged. The reference
estimator differentiates from the fourth pre-norm boundary back to the first,
second, and third boundaries, including every intervening normalization and
recurrent block. This is not a Jacobian of a single physical stack traversal.
An analytic recurrent toy test verifies the full downstream derivative;
GPU checks require exact native output and extractor parity for both arms.

Fitting uses up to 32 seeded, separate WikiText-103 **training** prompts of 64
tokens. Reference-estimator positions 16 through 62 are used for fitting,
with cotangents summed over valid future target positions and averaged over
valid source positions. This fitting convention does not change the audit's
first 64 consecutive positions on 1,024 **validation** documents.
Per-prompt Jacobians and cumulative means are saved in float32, along with
split-half and running-mean diagnostics. This is a pilot fit; a small
running-mean update alone does not establish convergence.

The toolkit JLensExtractor accepts optional zero-based `recurrence_idx` and
checks the lens boundary metadata. All four native passes execute, with all
adapters active in the organism. The selected activation is multiplied by the
fitted matrix, then projected with the final norm and LM head. This changes
the observational readout, not native inference or recurrent state updates.

`run.sh` is intended to run under the campaign's bounded process runner. A
maximum two-hour fitting budget leaves time for native parity tests and the
three standard audits inside the three-hour workload limit. The reference
package is loaded through PYTHONPATH to retain the pinned Ouro Transformers
runtime; its generic fitting API is used without changing its estimator.

## Lazy model device regression

The native parity checks exposed a general JLensExtractor initialization bug:
a lazily loaded base model still held meta parameters when the lens matrix's
device was selected. Copying a real Jacobian to meta discards its values;
a subsequent CUDA-by-meta matrix product can return meaningless values.
The extractor now dispatches model weights before choosing the lens device
and rejects any unresolved meta device. CPU regression tests check both
value preservation and rejection. The normal toolkit vocabulary projection
remains unchanged. These findings do not establish whether any other
project's previously saved J-lens results were affected.

The 128-prompt extension reuses the exact verified first 32 calibration
prompts and their per-prompt Jacobians. It retains the same 64-token context
and reference estimator. Cumulative snapshots are sparse; every per-prompt
matrix remains available to reconstruct intermediate means.
