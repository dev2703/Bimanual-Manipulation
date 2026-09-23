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

The separate handled serving-plate scene has a contact-only expert that passed
50/50 held-out lift, carry, release, and upright-placement trials at ±1.5 cm
object jitter. The original plain plate has not passed that gate. Watch or
rerun the handled-plate gate with:

```bash
.venv/bin/mjpython scripts/view_aloha_plate.py --seed 100000
.venv/bin/python scripts/aloha_plate_gate.py --episodes 50 --seed-offset 100000
```

Plate recordings use `--skill plate_pick_place` in
`bimanual.experts.generate_aloha_table` and replay with the same `--skill`
option in `bimanual.data.replay_aloha_table`. The plate scene and mug ACT scene
have separate artifact hashes; their data should not be mixed.

Once the local plate collection finishes, inspect it with
`make audit-plate replay-plate`. A matching validation split can be generated with:

```bash
.venv/bin/python -m bimanual.experts.generate_aloha_table \
  --skill plate_pick_place --split val --episodes 10
```

An isolated drawer scene now has clearance behind its physical handle. Its
contact-only expert passed 50/50 held-out open-and-release trials, with gripper
contact on the handle measured during each pull. The original mug scene's
near-flush handle does not pass that contact gate. Preview and retest it with:

```bash
.venv/bin/mjpython scripts/view_aloha_drawer.py --seed 100000
.venv/bin/python scripts/aloha_drawer_gate.py --episodes 50 --seed-offset 100000
```

Drawer recordings use `--skill drawer_open`; `make audit-drawer replay-drawer`
checks a completed training split. Keep each scene variant's dataset separate
until the physical geometry is consolidated and a combined task is revalidated.

The dinner dataset writer finalizes completed episodes individually. To
continue a cleanly interrupted collection, rerun it with the same skill,
split, root, and episode target plus `--resume`. For example, after stopping
the plate collection:

```bash
.venv/bin/python -m bimanual.experts.generate_aloha_table \
  --skill plate_pick_place --split train --episodes 50 --resume
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
