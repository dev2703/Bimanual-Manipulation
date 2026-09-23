.PHONY: setup test demo gen-data mug-gate mug-demo audit-mug replay-mug plate-gate plate-demo audit-plate replay-plate drawer-gate drawer-demo audit-drawer replay-drawer clean

UV := uv

setup:
	$(UV) sync

test:
	$(UV) run pytest tests -q

demo:
	$(UV) run python scripts/smoke_render.py

gen-data:
	$(UV) run python -m bimanual.experts.generate

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

clean:
	rm -rf .pytest_cache **/__pycache__ outputs/*.mp4
