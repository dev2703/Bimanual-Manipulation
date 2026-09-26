"""Trace the right-gripper grasp of the mug for a policy or the scripted expert.

Per 10 Hz step it logs the commanded and measured right-gripper opening, the
servo squeeze (kp * (opening - command), positive when closing on an object),
right-finger/mug contact count, and mug height. Comparing a learned policy to
the expert on the same seed separates a weak grip command from a bad grasp pose.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from bimanual.experts.aloha_mug import run_mug_pick_place
from bimanual.policy.aloha_act_runner import run_aloha_act_episode
from bimanual.sim.aloha_env import AlohaTableSettingEnv

GRIPPER = 13
FINGER_BODIES = ("right/left_finger_link", "right/right_finger_link")


class GraspTrace:
    def __init__(self, env: AlohaTableSettingEnv) -> None:
        self.env = env
        model = env.model
        self.mug = model.body("mug").id
        self.gripper_site = model.site("right/gripper").id
        self.fingers = {model.body(name).id for name in FINGER_BODIES}
        self.kp = float(model.actuator_gainprm[GRIPPER][0])
        self.rows: list[dict] = []
        self.control_ticks = 0

    def finger_mug_contacts(self) -> int:
        model, data = self.env.model, self.env.data
        count = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            count += self.mug in bodies and bool(bodies & self.fingers)
        return count

    def __call__(self) -> None:
        self.control_ticks += 1
        if self.control_ticks % 3:
            return
        data = self.env.data
        command = float(data.ctrl[GRIPPER])
        opening = float(self.env.state_vector()[GRIPPER])
        self.rows.append({
            "step": len(self.rows) + 1,
            "command": command,
            "opening": opening,
            "squeeze_n": self.kp * (opening - command),
            "contacts": self.finger_mug_contacts(),
            "mug_z": float(data.xpos[self.mug][2]),
            "gripper_minus_mug": (data.site_xpos[self.gripper_site] - data.xpos[self.mug]).round(4).tolist(),
            "mug_upright": self.env.mug_upright_cosine(),
            "right_joints": self.env.state_vector()[7:13].round(4).tolist(),
        })


def load_policy(kind: str, checkpoint: str, device: str):
    if kind == "smolvla":
        from bimanual.policy.smolvla.runner import MUG_INSTRUCTION, load_smolvla_bundle
        return (*load_smolvla_bundle(checkpoint, device), MUG_INSTRUCTION)
    from bimanual.policy.act_runner import load_act_bundle
    return (*load_act_bundle(checkpoint, device), None)


def summarize(rows: list[dict]) -> dict:
    gripping = [r for r in rows if r["contacts"] > 0]
    initial_z = rows[0]["mug_z"] if rows else float("nan")
    return {
        "steps": len(rows),
        "gripping_steps": len(gripping),
        "min_command": min((r["command"] for r in rows), default=None),
        "median_command_while_gripping": float(np.median([r["command"] for r in gripping])) if gripping else None,
        "median_squeeze_n_while_gripping": float(np.median([r["squeeze_n"] for r in gripping])) if gripping else None,
        "median_opening_while_gripping": float(np.median([r["opening"] for r in gripping])) if gripping else None,
        "peak_lift_m": max((r["mug_z"] for r in rows), default=initial_z) - initial_z,
        "grasp_offset_at_first_contact": gripping[0]["gripper_minus_mug"] if gripping else None,
        "min_upright_cosine": min((r["mug_upright"] for r in rows), default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["expert", "smolvla", "act"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[200002])
    parser.add_argument("--max-policy-steps", type=int, default=240)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.policy != "expert" and not args.checkpoint:
        parser.error("--checkpoint is required for learned policies")

    bundle = load_policy(args.policy, args.checkpoint, args.device) if args.policy != "expert" else None
    report = {}
    for seed in args.seeds:
        env = AlohaTableSettingEnv()
        env.reset(seed=seed, randomize_objects=True)
        trace = GraspTrace(env)
        try:
            if bundle is None:
                env.after_step = trace
                success = run_mug_pick_place(env).success
            else:
                policy, preprocessor, postprocessor, instruction = bundle
                torch.manual_seed(seed)
                success = run_aloha_act_episode(
                    env, policy, preprocessor=preprocessor, postprocessor=postprocessor,
                    device=args.device, max_policy_steps=args.max_policy_steps,
                    task="mug_pick_place", instruction=instruction, after_control_step=trace,
                ).success
        finally:
            env.close()
        report[str(seed)] = {"success": success, "summary": summarize(trace.rows), "trace": trace.rows}
        print(json.dumps({"policy": args.policy, "seed": seed, "success": success, **summarize(trace.rows)}), flush=True)

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"policy": args.policy, "checkpoint": args.checkpoint, "seeds": report}, handle, indent=2)


if __name__ == "__main__":
    main()
