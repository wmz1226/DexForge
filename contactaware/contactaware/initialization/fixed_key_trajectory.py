"""Palm targets between keys: the MANO palm composed with interpolated key palm offsets."""

from dataclasses import dataclass

import numpy as np

from contactaware.contact.object_model import world_to_obj
from contactaware.initialization.palm_motion import blended_pose
from contactaware.solver.palm import human_palm_frame
from contactaware.solver.interpolation import position_interpolator, rotation_interpolator
from contactaware.solver.pose import fit_palm_pose


@dataclass(frozen=True)
class KeyPalmObjective:
    local_positions: np.ndarray
    local_rotations: np.ndarray

    def target(self, problem):
        return self.target_at(problem.hand, problem.joints, problem.frame_id)

    def target_at(self, hand, joints, frame):
        position, rotation = human_palm_frame(joints, hand.profile.track_finger_ids)
        return (position + rotation @ self.local_positions[frame],
                rotation @ self.local_rotations[frame])

    def initial_states(self, inputs, frames, states, ratio, *, interpolated, confidence):
        """Follow demonstration in free motion and solved grasp geometry in contact."""
        hand = inputs.hand
        output = np.array(interpolated, copy=True)
        previous = states[0, :hand.base_qpos_dim]
        for frame, state in enumerate(output):
            joints = world_to_obj(inputs.mano_joints[frame], inputs.obj_pose[frame])
            demonstrated = self.target_at(hand, joints, frame * ratio)
            grasp_position, grasp_rotation, _, _ = hand.palm_pose_jacobian(state)
            target = blended_pose(demonstrated, (grasp_position, grasp_rotation), confidence[frame])
            guess = np.r_[previous, state[hand.base_qpos_dim:]]
            output[frame] = fit_palm_pose(hand, guess, *target)
            previous = output[frame, :hand.base_qpos_dim]
        output[frames] = states
        return output

    @classmethod
    def from_keys(cls, inputs, frames, states, ratio):
        positions, rotations = [], []
        for frame, state in zip(frames, states):
            joints = world_to_obj(inputs.mano_joints[frame], inputs.obj_pose[frame])
            hp, hr = human_palm_frame(joints, inputs.hand.profile.track_finger_ids)
            rp, rr, _, _ = inputs.hand.palm_pose_jacobian(state)
            positions.append(hr.T @ (rp - hp))
            rotations.append(hr.T @ rr)
        times = np.arange((len(inputs.mano_joints) - 1) * ratio + 1) / ratio
        if len(frames) == 1:
            return cls(np.repeat(positions, len(times), axis=0), np.repeat(rotations, len(times), axis=0))
        times = np.clip(times, frames[0], frames[-1])
        position = position_interpolator(frames, positions)(times)
        rotation = rotation_interpolator(frames, rotations)(times)
        return cls(position, rotation)
