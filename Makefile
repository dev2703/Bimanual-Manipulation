.PHONY: setup test demo gen-data mug-gate mug-demo audit-mug replay-mug plate-gate plate-demo audit-plate replay-plate drawer-gate drawer-demo audit-drawer replay-drawer verify-data clean

UV := uv

setup:
	$(UV) sync

test:
	$(UV) run pytest tests -q

demo:
	$(UV) run python scripts/render_aloha_expert.py --task combined --seed 0

gen-data:
	$(UV) run python -m bimanual.experts.generate_aloha_table --skill mug_pick_place --episodes 1 --root outputs/aloha_gen_data_smoke --overwrite

mug-gate:
	$(UV) run python scripts/aloha_mug_gate.py --episodes 50

mug-demo:
	.venv/bin/mjpython scripts/view_aloha_mug.py --seed 100000

audit-mug:
	$(UV) run python -m bimanual.data.audit outputs/aloha_mug_train

replay-mug:
	$(UV) run python -m bimanual.data.replay_aloha_table outputs/aloha_mug_train

plate-gate:
	$(UV) run python scripts/aloha_plate_gate.py --episodes 50 --seed-offset 100000

plate-demo:
	.venv/bin/mjpython scripts/view_aloha_plate.py --seed 100000

audit-plate:
	$(UV) run python -m bimanual.data.audit outputs/aloha_plate_train

replay-plate:
	$(UV) run python -m bimanual.data.replay_aloha_table outputs/aloha_plate_train --skill plate_pick_place

drawer-gate:
	$(UV) run python scripts/aloha_drawer_gate.py --episodes 50 --seed-offset 100000

drawer-demo:
	.venv/bin/mjpython scripts/view_aloha_drawer.py --seed 100000

audit-drawer:
	$(UV) run python -m bimanual.data.audit outputs/aloha_drawer_train

replay-drawer:
	$(UV) run python -m bimanual.data.replay_aloha_table outputs/aloha_drawer_train --skill drawer_open

verify-data:
	$(UV) run python -m bimanual.data.verify_index

SKILL ?= mug_pick_place
EPISODES ?= 10
gate:
	$(UV) run python scripts/gate.py --skill $(SKILL) --episodes $(EPISODES)

audit:
	$(UV) run python -m bimanual.data.audit outputs/aloha_$$($(UV) run python -c "from bimanual.skills.registry import get_skill; print(get_skill('$(SKILL)').bucket)")_train

replay:
	$(UV) run python -m bimanual.data.replay_aloha_table outputs/aloha_$$($(UV) run python -c "from bimanual.skills.registry import get_skill; print(get_skill('$(SKILL)').bucket)")_train --skill $(SKILL)

clean:
	rm -rf .pytest_cache **/__pycache__ scripts/__pycache__ outputs/*.mp4 outputs/*_smoke outputs/*_smoke.log
