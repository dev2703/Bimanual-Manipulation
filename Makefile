.PHONY: help setup test skills gate dinner-gate pour-gate view scene demo gen-data audit replay verify-data clean

UV := uv
MJPYTHON := .venv/bin/mjpython
SKILL ?= mug_pick_place
EPISODES ?= 50
SEED ?= 100000
BUCKET = $(shell $(UV) run python -c "from bimanual.skills.registry import get_skill; print(get_skill('$(SKILL)').bucket)")
DATA ?= outputs/aloha_$(BUCKET)_train

help:  ## List targets. Most take SKILL=<name> (see `make skills`).
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install the locked environment.
	$(UV) sync

test:  ## Run the full test suite.
	$(UV) run pytest tests -q

skills:  ## List registered skills and their gate status.
	$(UV) run python scripts/gate.py --list

gate:  ## Held-out scripted-expert gate: make gate SKILL=plate_pick_place EPISODES=50
	$(UV) run python scripts/gate.py --skill $(SKILL) --episodes $(EPISODES)

dinner-gate:  ## Full dinner sequence gate (drawer, plate, cutlery, mug, handoff, pour).
	$(UV) run python scripts/aloha_combined_sequence_gate.py --episodes $(EPISODES)

pour-gate:  ## Standalone pour gate with action-only replay.
	$(UV) run python scripts/aloha_pour_replay_probe.py --episodes $(EPISODES) --output outputs/gates/pour_stream_pose_50.json

view:  ## Watch the scripted expert: make view SKILL=drawer_open SEED=100000
	$(MJPYTHON) scripts/view.py --skill $(SKILL) --seed $(SEED)

scene:  ## Open a skill's scene without acting.
	$(MJPYTHON) scripts/view.py --skill $(SKILL) --seed $(SEED) --scene-only

demo:  ## Render the combined expert to video.
	$(UV) run python scripts/render_aloha_expert.py --task combined --seed 0

gen-data:  ## Record a one-episode smoke dataset for SKILL.
	$(UV) run python -m bimanual.data.record --skill $(SKILL) --episodes 1 --root outputs/smoke/aloha_$(BUCKET) --overwrite

audit:  ## Audit a recorded dataset (DATA defaults to outputs/aloha_<bucket>_train).
	$(UV) run python -m bimanual.data.audit $(DATA)

replay:  ## Replay a recorded dataset's actions and check success.
	$(UV) run python -m bimanual.data.replay $(DATA) --skill $(SKILL)

verify-data:  ## Check datasets against DATA_INDEX.json.
	$(UV) run python -m bimanual.data.verify_index

clean:  ## Remove caches and smoke outputs.
	rm -rf .pytest_cache **/__pycache__ scripts/__pycache__ outputs/*.mp4 outputs/*_smoke outputs/*_smoke.log
