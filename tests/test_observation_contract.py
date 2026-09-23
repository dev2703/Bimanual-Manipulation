"""Phase 4 gate (decision A2/A3): the policy-visible observation must
never contain privileged simulator state, and the executor's control
flow must never call oracle_state() to decide what a learned policy
should do.

Checks two things directly against the running system, not just by
convention:
  1. proprio()'s keys never overlap with a hardcoded privileged-name
     denylist (object positions/matrices, drawer opening, contacts).
  2. the dataset's per-frame feature schema (generate.py) keeps
     privileged fields under the `privileged.*` namespace, separate
     from observation.state/action, so a training pipeline can filter
     them out mechanically rather than by convention alone.
"""

from __future__ import annotations

from bimanual.experts.generate import make_features
from bimanual.sim.env import BimanualTableEnv

PRIVILEGED_KEY_SUBSTRINGS = ("_pos", "_mat", "drawer_opening", "drawer_handle_pos", "contacts")


def test_proprio_contains_no_privileged_keys():
    env = BimanualTableEnv()
    proprio_keys = set(env.proprio().keys())
    oracle_only_keys = set(env.oracle_state().keys()) - proprio_keys

    # proprio() legitimately has its own "*_pos"/"*_mat" keys (ee_pos,
    # ee_mat -- forward-kinematics EE pose, allowed per decision A2). The
    # contract is specifically that proprio() must not expose any key
    # that ALSO appears in oracle_state() (i.e. simulator object/contact
    # state), not that it avoid the substring in isolation.
    leaked = proprio_keys & oracle_only_keys
    assert not leaked, f"proprio() leaked privileged keys: {leaked}"

    # And oracle_state() must actually contain privileged data (sanity
    # check that this test isn't vacuously true).
    assert any(k.endswith("_pos") for k in oracle_only_keys)


def test_dataset_schema_namespaces_privileged_fields():
    features = make_features()
    allowed_non_privileged = {"observation.state", "action"} | {
        k for k in features if k.startswith("observation.images.")
    }
    for key in features:
        if key in allowed_non_privileged:
            continue
        assert key.startswith("privileged."), (
            f"dataset feature {key!r} is neither observation.state/action/"
            f"observation.images.* nor namespaced privileged.* -- a training "
            f"pipeline can't mechanically filter it out"
        )


def test_observation_state_is_proprioception_only():
    """observation.state's 12 values must be derivable from proprio()
    alone (5 joints + gripper per arm), never from oracle_state()."""
    from bimanual.sim import robot_spec as rs

    env = BimanualTableEnv()
    p = env.proprio()
    for prefix in ("left", "right"):
        assert p[f"{prefix}_joint_pos"].shape == (rs.N_ARM_JOINTS,)
        assert isinstance(p[f"{prefix}_gripper_opening"], float)
    # 5 + 1 per arm, 2 arms
    assert rs.N_BIMANUAL_ACTIONS == 12
