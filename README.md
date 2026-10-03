# Bimanual dinner-table manipulation (MuJoCo, ALOHA 2)

A simulated bimanual dinner table set up problem:
open a drawer, place the plate, fork, spoon and glass, hand an object between
arms, and pour into the glass. Every skill has a scripted contact-based expert
with a held-out gate; those experts produce the datasets for learned policies
(ACT, SmolVLA).

## Quick start

```bash
make setup          # uv sync
make test           # full test suite
make skills         # registered skills and gate status
make view SKILL=drawer_open           # watch an expert (macOS: uses mjpython)
make gate SKILL=plate_pick_place      # 50-episode held-out gate -> outputs/gates/
make dinner-gate                      # full dinner sequence gate
make help           # every target
```

Rendering tests need an OpenGL context; run them outside restricted sandboxes.

## Status

| Gate | Result | Notes |
|---|---|---|
| Drawer, plate, mug, bottle, block, plate recovery | marked passed in the registry | earlier 50-episode gates; rerun with `make gate SKILL=...` |
| Fork and spoon | 50/50 | drawer is opened as an unrecorded setup step |
| Baton handoff | 50/50 | carrier alternates left/right by seed |
| Pour | 50/50 | ballistic stream proxy, see below |
| Dinner sequence | 50/50 on two seed sets | drawer, plate, fork, spoon, mug, handoff, pour |

## Layout

| Path | Contents |
|---|---|
| `bimanual/skills/registry.py` | One `Skill` per task: scene, expert, instruction, success check |
| `bimanual/experts/` | Scripted contact experts (`aloha_*.py`) |
| `bimanual/evaluation/` | Gates, success predicates, pour geometry, composed eval |
| `bimanual/sim/`, `bimanual/control/` | MuJoCo environments, IK |
| `bimanual/policy/`, `bimanual/training/` | Policy adapters, training configs, ROCm launcher |
| `bimanual/data/` | Dataset recording, audit, replay, `DATA_INDEX.json` verification |
| `scripts/` | CLIs: `gate.py`, `view.py`, `eval_policy.py`, sequence and pour gates |
| `assets/robots/aloha/` | Scenes (`task_*.xml`) |

To add a skill: write an expert `run(env, record_frames=False)` returning a
result with `.success`, register it in `bimanual/skills/registry.py`, then run
`make gate SKILL=<name>`.

## Datasets

Recordings are written per skill and finalize each episode individually:

```bash
.venv/bin/python -m bimanual.data.record \
  --skill plate_pick_place --split train --episodes 50 [--resume]
make audit replay SKILL=plate_pick_place
```


## Training and evaluation reference

For the dinner ACT baseline, audit and replay the generated mug data, then
train with the fixed 20-step chunk and 8-step execution prefix:

```bash
make audit replay SKILL=mug_pick_place
.venv/bin/lerobot-train --config_path=bimanual/training/configs/act_aloha_mug.yaml
```

Run a trained dinner-task ACT checkpoint with the interactive MuJoCo viewer:

```bash
.venv/bin/mjpython scripts/view.py --skill mug_pick_place --seed 100000 \
  --checkpoint outputs/act_aloha_mug_full/checkpoints/020000/pretrained_model
```

The checkpoint above has completed 20,000 of the planned 100,000 updates. It
placed the mug on 13/20 held-out trials (65%) in the physical scene, clearing
the 50% mug ACT baseline gate for this seed set. Training can resume from the
saved optimizer state with:

```bash
.venv/bin/lerobot-train \
  --config_path=outputs/act_aloha_mug_full/checkpoints/020000/pretrained_model/train_config.json \
  --resume=true
```

## Problem statement


Problem: an end-to-end simulated Physical AI system for bimanual manipulation for setting up a dinner table.

The objective is to build a reproducible Physical AI system that can take a natural-language instruction such as:

“Open the top drawer, retrieve the fork and spoon, put the plate in the centre, place the fork and spoon beside it, place the mug on the right, and pour water into the mug.”

and execute the full task in MuJoCo using two ALOHA 2 arms. SO-101 remains a
legacy comparison profile.

The challenge is not simply pick-and-place. The system must combine:

natural-language task understanding;

visual scene grounding;

temporal memory across a long horizon;

action-chunk generation;

dynamic coordination of two arms;

contact-rich manipulation;

handoffs and complementary dual-arm actions;

progress and success estimation;

closed-loop replanning;

recovery after unexpected changes;

generalization across scene, object, lighting, and physics variations;

deterministic, reproducible evaluation.

The architecture should therefore not be a monolithic language + image -> 12-D joint trajectory policy. Nor should it be a conventional LLM planner that converts language into a fixed list of scripted skills and then ignores the scene.
