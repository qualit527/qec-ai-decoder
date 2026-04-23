# Demo 2 Runtime Notes

## CPU

Fast mode is the recommended smoke path: one dev-profile round on BB72. On the
sample run committed in `expected_output/sample_run/`, the round-level metrics
show `train_wallclock_s = 0.3976` and `eval_wallclock_s = 0.0728`. End-to-end
CLI wall clock is slightly higher because it includes process startup and
artifact writing.

`MODE=dev` runs three dev-profile rounds and is the best choice when you want a
small multi-round demo without waiting for prod budgets.

## GPU

This repository can train the neural predecoder with GPU-backed torch, but the
BB72 smoke loop is already short in dev mode. GPU helps more when you scale up
round count or move to `PROFILE=prod`; it does not change the artifact contract
or the OSD routing.

## OSD Cost

The BB72 demo is intentionally different from the surface-code demo because the
classical backend is OSD, not MWPM. OSD is the relevant qLDPC routing in
`autoqec/envs/builtin/bb72_depol.yaml`, and its cost stays on the classical
decode side. In practice that means:

- dev smoke runs are cheap enough for a quick demo;
- prod-style runs get more expensive faster than the surface/MWPM path; and
- if runtime spikes, OSD evaluation is the first place to look before blaming
  the demo wrapper.
