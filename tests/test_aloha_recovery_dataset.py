import numpy as np

from bimanual.experts.generate_aloha_table import make_features, write_episode


class _Dataset:
    def __init__(self):
        self.frames = []
        self.saved = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved = True


def test_recovery_schema_marks_one_event_after_approach():
    features = make_features(recovery=True)
    assert "privileged.failure_event" in features
    assert "privileged.failure_event" not in make_features()
    dataset = _Dataset()
    observation = {"observation.state": np.zeros(14, dtype=np.float32),
                   "observation.velocity": np.zeros(14, dtype=np.float32)}
    oracle = {f"{name}_pos": np.zeros(3) for name in ("plate", "mug", "bottle", "fork", "spoon")}
    oracle["drawer_opening"] = np.zeros(1)
    frames = {name: np.zeros((2, 2, 3), dtype=np.uint8)
              for name in ("overhead_cam", "wrist_cam_left", "wrist_cam_right")}
    record = [dict(observation=observation, oracle=oracle, frames=frames,
                   action=np.ones(14), phase=phase, arm="right", timestamp=0.1 * i)
              for i, phase in enumerate(("APPROACH", "APPROACH", "PRE_GRASP", "GRASP"))]
    write_episode(dataset, record, "Place the plate", 5, recovery=True,
                  perturbation_xy=np.array([0.02, -0.02]))
    assert dataset.saved
    assert [float(f["privileged.failure_event"][0]) for f in dataset.frames] == [0, 0, 1, 0]
    assert [float(f["privileged.recovery_active"][0]) for f in dataset.frames] == [0, 0, 1, 1]
    assert all(float(f["privileged.recovery_outcome"][0]) == 1 for f in dataset.frames)
    assert all(np.allclose(f["privileged.perturbation_xy"], [0.02, -0.02]) for f in dataset.frames)
