"""Initialize some searches with the model's velocity-compensated control."""
import numpy as np

from forceaware.optimizer import WindowOptimizer


def velocity_prior(targets, lag, knot_dt, mpc_dt):
    if len(targets.actions) > 1:
        derivative = np.gradient(targets.actions.astype(float), knot_dt, axis=0)
    elif len(targets.dense_actions) > 1:
        derivative = np.gradient(
            targets.dense_actions.astype(float), mpc_dt, axis=0
        )[-1:]
    else:
        derivative = np.zeros_like(targets.actions, dtype=float)
    return derivative * lag


class ServoPriorOptimizer(WindowOptimizer):
    def __init__(self, cpu_model, device_model, index, **kwargs):
        super().__init__(cpu_model, device_model, index, **kwargs)
        actuator = index.hand_ctrl
        self.servo_lag = (
            -cpu_model.actuator_biasprm[actuator, 2]
            / cpu_model.actuator_gainprm[actuator, 0]
        )
        grid = kwargs['cfg'].simulator.time_grid
        self.knot_dt, self.mpc_dt = grid.knot_dt, grid.mpc_dt

    def _prepare_window(self, qpos, qvel, targets, **kwargs):
        super()._prepare_window(qpos, qvel, targets, **kwargs)
        prior = velocity_prior(
            targets, self.servo_lag, self.knot_dt, self.mpc_dt
        ).reshape(-1)
        raw = self.control.raw.numpy()
        dimension = self.index.action_dim
        keep = max(1, self.settings.worlds // 2)
        raw[keep:, dimension:] += prior - kwargs['warm'][dimension:]
        self.control.raw.assign(raw)
        self.control.best_raw.assign(raw)
        self.control.last_evaluated_raw.assign(raw)
