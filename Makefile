.PHONY: setup test demo gen-data mug-gate mug-demo audit-mug replay-mug clean

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

clean:
	rm -rf .pytest_cache **/__pycache__ outputs/*.mp4
