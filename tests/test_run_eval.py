from pathlib import Path

from bimanual.evaluation.run_eval import run_episode, write_report
from bimanual.training.rocm.launch import load_job, smoke


class _Result:
    def __init__(self, success: bool) -> None:
        self.success = success


def test_episode_record_stops_on_the_first_failure(tmp_path: Path):
    calls = []

    def ok(env):
        calls.append(env)
        return _Result(True)

    def bad(env):
        calls.append(env)
        return _Result(False)

    def later(env):
        raise AssertionError("should not run")

    episode = run_episode(object(), [("drawer", ok), ("plate", bad), ("mug", later)], "Set the table", seed=3)
    assert episode["success"] is False
    assert episode["completed"] == ["drawer"]
    assert "Done: nothing yet" in episode["stages"][0]["instruction"]
    report = write_report(tmp_path / "eval.json", [episode], threshold=0.7)
    assert report["passed"] is False
    assert report["stage_successes"] == {"drawer": 1}


def test_job_smoke_runs_the_configured_step_count(tmp_path: Path):
    job = load_job("smolvla_memory")
    assert job["instruction"] == "memory"
    assert job["n_action_steps"] == 20
    report = smoke(job, tmp_path / "smoke.json")
    assert report["smoke_steps"] == 20
    assert report["loss_decreased"]
