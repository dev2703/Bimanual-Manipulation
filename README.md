# Bimanual dinner-table manipulation (MuJoCo, ALOHA 2)

A simulated Physical AI system in which two ALOHA 2 arms set a dinner table:
open a drawer, place the plate, fork, spoon and glass, hand an object between
arms, and pour into the glass. Every skill has a scripted contact-based expert
with a held-out gate; those experts produce the datasets for learned policies
(ACT, SmolVLA). SO-101 code is archived under `bimanual/legacy_so101`.

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
| Fork and spoon (`scripts/aloha_cutlery_gate.py`) | 50/50 | |
| Baton handoff | 50/50 | carrier alternates left/right by seed |
| Pour | 50/50 | ballistic stream proxy, see below |
| Dinner sequence | 50/50 on two seed sets | drawer, plate, fork, spoon, mug, handoff, pour |

The pour metric checks that a 4 mm stream leaving the bottle's lower lip at
0–0.2 m/s lands inside the glass opening for every sampled exit speed, with
the glass upright and no finger–glass contact, for 20 consecutive steps. No
liquid is simulated (`bimanual/evaluation/pour_geometry.py`). The dinner
sequence still uses scripted staging between stages: neutral arm resets and
baton repositioning.

Not done yet: audited 37-D datasets for every skill, per-skill ACT baselines,
the 14-D vs 37-D ablation, and a learned end-to-end dinner policy.

## Layout

| Path | Contents |
|---|---|
| `bimanual/skills/registry.py` | One `Skill` per task: scene, expert, instruction, success check |
| `bimanual/experts/` | Scripted contact experts (`aloha_*.py`) and dataset generation |
| `bimanual/evaluation/` | Gates, success predicates, pour geometry, composed eval |
| `bimanual/sim/`, `bimanual/control/` | MuJoCo environments, IK |
| `bimanual/policy/`, `bimanual/training/` | Policy adapters, training configs, ROCm launcher |
| `bimanual/data/` | Dataset audit, replay, `DATA_INDEX.json` verification |
| `scripts/` | CLIs: `gate.py`, `view.py`, `eval_policy.py`, sequence and pour gates |
| `assets/robots/aloha/` | Scenes (`task_*.xml`) |

To add a skill: write an expert `run(env, record_frames=False)` returning a
result with `.success`, register it in `bimanual/skills/registry.py`, then run
`make gate SKILL=<name>`.

## Datasets

Recordings are written per skill and finalize each episode individually:

```bash
.venv/bin/python -m bimanual.experts.generate_aloha_table \
  --skill plate_pick_place --split train --episodes 50 [--resume]
make audit replay SKILL=plate_pick_place
```

Each skill's scene has its own artifact hash; keep datasets from different
scenes separate when training.

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

The physical ALOHA phase-5 path now has a verifier-driven goal loop, a
three-camera RGB predicate model, and a separate held-out evaluator. Training
on ordinary expert frames alone gave a misleading perfect frame score: moving
the mug after the robot finished did not change its answer. Matched images
with the same robot pose and a moved mug corrected that shortcut. The revised
verifier reached 100% recall for each class on all 2,170 frames from 10
held-out episodes, flipped correctly on 10/10 live moved-mug tests, and drove
the scripted mug expert to 10/10 correct goal transitions. Plate and drawer
verifiers reached 99.92% and 99.57% agreement, respectively, across all 2,540
plate and 1,840 drawer held-out frames, as well as 10/10 matched counterfactual
pairs and 10/10 live perturbation flips each. Their three-model ensemble drove
the scripted drawer–plate–mug sequence to 10/10 correct goal transitions in the
combined scene and flipped each prediction after moving its object or closing
the drawer in all 10 trials. This passes the current three-goal scripted-expert
phase-5 verifier gate; cutlery and pouring are not included.
Generate the matched images and reproduce the training and live gate with:

```bash
.venv/bin/python scripts/collect_aloha_verifier_pairs.py --episodes 50 \
  --output outputs/aloha_mug_verifier_pairs_train.npz
.venv/bin/python scripts/collect_aloha_verifier_pairs.py --episodes 10 \
  --seed-offset 100000 --output outputs/aloha_mug_verifier_pairs_val.npz
HF_HOME=outputs/.cache/hf .venv/bin/python -m bimanual.training.train_aloha_verifier \
  --skill mug_pick_place --train-root outputs/aloha_mug_train \
  --val-root outputs/aloha_mug_val \
  --train-counterfactual outputs/aloha_mug_verifier_pairs_train.npz \
  --val-counterfactual outputs/aloha_mug_verifier_pairs_val.npz \
  --output outputs/aloha_mug_verifier_grounded.pt
.venv/bin/python scripts/probe_aloha_verifier_counterfactual.py \
  outputs/aloha_mug_verifier_grounded.pt
.venv/bin/python scripts/eval_aloha_verified_executor.py \
  outputs/aloha_mug_verifier_grounded.pt --episodes 10
.venv/bin/python scripts/eval_aloha_combined_verifier.py \
  --drawer outputs/aloha_drawer_verifier_grounded.pt \
  --plate outputs/aloha_plate_verifier_grounded.pt \
  --mug outputs/aloha_mug_verifier_grounded.pt --episodes 10
```

An opt-in 37-D cooperative ALOHA state adds relative gripper pose, both EE
twists, and gripper openings. Existing 14-D ACT checkpoints retain their
original state contract. For recovery, a 3.5 cm plate move after approach yielded 50/50 physical
expert successes when the grasp was replanned and 0/50 with a stale grasp:

```bash
.venv/bin/python scripts/aloha_plate_recovery_gate.py --episodes 50
.venv/bin/python -m bimanual.experts.generate_aloha_table \
  --skill plate_recovery --episodes 50
.venv/bin/python -m bimanual.data.replay_aloha_table \
  outputs/aloha_plate_recovery_train --skill plate_recovery
```

A three-episode recovery recording was audited and replayed successfully,
including the exact exogenous displacement at its recorded event frame.
The full 50-episode recovery training split and 10-episode validation split
were also audited and replayed successfully. Recovery policy training and the
matched with/without-recovery comparison remain open.

The SmolVLA adapter loads saved LeRobot processors and uses the same ALOHA
rollout logic as ACT. The Modal smoke job is defined in
`bimanual/training/modal_app.py`; it expects an audited 50-episode mug dataset
in a private `bimanual-dinner` volume and does not contain tokens. A successful
SmolVLA checkpoint and the plain-instruction versus serialized-memory
comparison have **not** been produced yet.

For an AMD GPU droplet, `bimanual/training/rocm/` provides a separate SmolVLA
training container with PyTorch 2.11 ROCm 7.2 wheels. It uses the droplet's
local `rocm:latest` image (Python 3.12) as its base, pins the installed ROCm
torch/torchvision versions while installing LeRobot, and requires a working
AMD GPU before training. This standalone training container does not install
the Python 3.13 simulation project.

Place `train.py`, `run.sh`, and the audited `datasets/aloha_mug_train` under
`/root/bimanual-training` on the droplet. Build the Dockerfile there, then run:

```bash
docker build -t bimanual-smolvla:rocm /root/bimanual-training/build
bash /root/bimanual-training/run.sh 200 smolvla-mug-smoke
docker logs -f smolvla-mug-smoke
# After the smoke container exits successfully and saves its checkpoint:
bash /root/bimanual-training/run.sh 20000 smolvla-mug-20k
docker logs -f smolvla-mug-20k
```

These detached containers survive SSH disconnection. Checkpoints and the
model cache persist under `/root/bimanual-training`; the longer run saves
every 1,000 steps. Hub uploads and W&B are disabled. The 20,000-step run starts
from the pretrained SmolVLA base independently of the smoke run. GPU access
uses `/dev/kfd` and `/dev/dri`; PyTorch's device name remains `cuda` on ROCm.
The launcher infers camera/state/action features from the dataset and builds
an `aloha_mug_train_stable_stats` training copy. It recomputes state/action
mean and standard deviation in float64 with a `1e-4` standard-deviation floor:
the original float32 metadata reports zero variance for some nearly constant
joints and otherwise produces extreme normalized values. Source recordings
remain intact, and checkpoints save the corrected normalization processors.

The bounded droplet experiment tools also include a 159.85M-parameter custom
preset (`train_compact_vla --config research160`) with optional disjoint
held-out validation and periodic resumable checkpoint contents. This is an
action-only feasibility experiment; its auxiliary heads are not supervised.
`rocm/evaluate.py` compares first-action errors on 80 held-out expert frames;
those metrics are not closed-loop task success rates.

`rocm/finish_experiments.py` waits for the named SmolVLA and compact jobs with
a 50-minute limit, runs a bounded evaluation, and stages the latest checkpoints,
logs, source, and metadata with a SHA-256 manifest. Run
`rocm/backup_and_poweroff.py` locally to download that export, verify every
file, and request droplet power-off only after the backup succeeds. Keep the
local process and network connection alive until it writes `BACKUP_VERIFIED`.
The separate 90-minute systemd failsafe powers off even if the local backup
process is unavailable; remote data remains on disk in that case.
**Powering off does not end DigitalOcean billing. Delete the droplet through
the dashboard or authenticated API after verifying the local backup.**

For a private Hugging Face backup, authenticate locally with
`.venv/bin/hf auth login` (write access), then run
`rocm/publish_huggingface.py --project <project-path> --artifacts <backup-path>`.
It uploads the original train/validation datasets, latest SmolVLA weights and
training state, the compact model and optimizer checkpoint, and experiment
metrics/source/normalization metadata into separate private repositories in
the authenticated account. Repo names end in `603294423` for this experiment.
It waits for `BACKUP_VERIFIED`, validates remote SHA-256/Git blob hashes, and
writes `HUGGINGFACE_VERIFIED` and `huggingface_status.json` beside the backup.
Credentials stay with the local Hub SDK. A local Mac notification is requested
after all uploads succeed; keep the watcher and network connection running.

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
