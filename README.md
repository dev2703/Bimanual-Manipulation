An experimental MuJoCo system for setting a dinner table with two ALOHA 2 arms.
The current physical learning task is released mug pick-and-place; the full
drawer, cutlery, plate, and pour sequence remains a research goal.

Preview the audience-facing ALOHA dinner-table scene on macOS:

```bash
.venv/bin/mjpython scripts/view_aloha_scene.py --seed 100000
```

Watch the first physical dinner skill set the mug at its place setting:

```bash
.venv/bin/mjpython scripts/view_aloha_mug.py --seed 100000
```

For the dinner ACT baseline, audit and replay the generated mug data, then
train with the fixed 20-step chunk and 8-step execution prefix:

```bash
make audit-mug replay-mug
.venv/bin/lerobot-train --config_path=bimanual/training/configs/act_aloha_mug.yaml
```

Run a trained dinner-task ACT checkpoint with the interactive MuJoCo viewer:

```bash
.venv/bin/mjpython scripts/view_aloha_act.py \
  outputs/act_aloha_mug/checkpoints/100000/pretrained_model \
  --task mug_pick_place --seed 100000 --device mps
```

The checkpoint path above is the expected output of the full mug ACT run, not a
checkpoint currently present in the repository. A short ACT learning probe
completed, but held-out mug success has not been established; the full
closed-loop dinner-task policy gate remains open.

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
