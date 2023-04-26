import copy

import io
import sys
sys.path.append('/home/yz685/low_rank_BOPE/low_rank_BOPE')
sys.path.append("/home/yz685/low_rank_BOPE/low_rank_BOPE/aux_software/spot_mini_mini")

from typing import Optional

import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem

from gym.wrappers import RecordVideo
from spotmicro.GaitGenerator.Bezier import BezierGait
from spotmicro.GymEnvs.spot_bezier_env import spotBezierEnv
from spotmicro.Kinematics.SpotKinematics import SpotModel
from spotmicro.OpenLoopSM.SpotOL import BezierStepper
from spotmicro.spot_env_randomizer import SpotEnvRandomizer


class spotBezierEnv2(spotBezierEnv):
    """
    Updates the environment for changes in the gym API in newer versions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_mode = "rgb_array"
        self.metadata["render_fps"] = 50

    def step(self, action):
        observations, reward, done, info = super().step(action)
        return observations, reward, done, False, info


def spot_mini_mini_trajectory(
    SwingPeriod: float = 0.2,
    StepVelocity: float = 0.001,
    ClearanceHeight: float = 0.05,
    roll: float = 0.0,
    pitch: float = 0.0,
    max_timesteps: int = 500,
    record_pos_every_n: int = 5,
    record: bool = False,
    results_path: str = None,
    name_prefix: str = "spot",
    seed: int = 1000
):
    r"""
    Run one trajectory of a spot mini mini robot (?)

    Args:

    Returns:
    """
    seed = seed
    StepLength = 0.05
    LateralFraction = 0.0
    YawRate = 0.0
    PenetrationDepth = 0.003
    yaw = 0.0
    orn = [roll, pitch, yaw]

    env = spotBezierEnv2(
        render=False,
        env_randomizer=SpotEnvRandomizer(),
        control_time_step=0.0,
    )
    dt = float(env._time_step)
    if record:
        assert results_path is not None
        env = RecordVideo(
            env=env,
            video_folder=results_path,
            name_prefix=name_prefix,
            step_trigger=lambda x: x >= 50,
        )

    # Set seeds
    env.seed(seed)
    np.random.seed(seed)

    state = env.reset()

    spot = SpotModel()
    T_bf0 = spot.WorldToFoot
    T_bf = copy.deepcopy(T_bf0)

    bzg = BezierGait(dt=dt)
    bzg.Tswing = SwingPeriod

    bz_step = BezierStepper(dt=dt, mode=0)
    bz_step.StepLength = StepLength
    bz_step.LateralFraction = LateralFraction
    bz_step.YawRate = YawRate
    bz_step.StepVelocity = StepVelocity

    action = env.action_space.sample()

    t = 0
    pos_trajectory = []
    while t < max_timesteps:
        bz_step.ramp_up()
        pos, _, _, _, _, _, _, _ = bz_step.StateMachine()

        contacts = state[-4:]
        # Get Desired Foot Poses
        T_bf = bzg.GenerateTrajectory(
            StepLength,
            LateralFraction,
            YawRate,
            StepVelocity,
            T_bf0,
            T_bf,
            ClearanceHeight,
            PenetrationDepth,
            contacts,
        )
        joint_angles = spot.IK(orn, pos, T_bf)

        env.pass_joint_angles(joint_angles.reshape(-1))
        # Get External Observations
        env.spot.GetExternalObservations(bzg, bz_step)
        # Step
        state = env.step(action)[0]
        if t % record_pos_every_n == 0:
            pos_trajectory.append(env.spot.GetBasePosition())
        t += 1
    env.close()

    # a list of tuples (x,y,z) indicating the position of the robot's centroid
    return pos_trajectory 


# Outcome function
class SpotMiniMiniProblem(BaseTestProblem):
    r"""
    Test problem class for spot mini mini robot
    """
    param_names = ["SwingPeriod", "StepVelocity", "ClearanceHeight", "roll", "pitch"]
    pi8 = np.pi / 8
    original_bounds = torch.tensor(
        [
            [0.1, 0.001, 0, -pi8, -pi8],
            [0.4, 3, 0.1, pi8, pi8],
        ]
    )
    noise_std = 0.0

    def __init__(
        self,
        dim: int = 3,
        max_timesteps: int = 500,
        record_pos_every_n: int = 5,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ):
        r"""
        Initialize the problem class. 
        Outcome is a flattened vector of the robot's centroid position (x,y,z)
        at each timestep, in the form [x1,x2,...,y1,y2,...,z1,z2,...].

        Args:
            dim: input dimension of the problem, must be <= 5.
                The idea is that we can control the first `dim` parameters in 
                ["SwingPeriod", "StepVelocity", "ClearanceHeight", "roll", "pitch"]
                in the simulator. All have default values if not specified.
            max_timesteps: maximum number of timesteps to run the simulation for
            record_pos_every_n: record the position every n timesteps
            noise_std: standard deviation of the noise to add to the objective
            negate: whether to negate the objective (if true, maximize) 
        """
        self.dim=dim
        if dim > 5: 
            raise ValueError("dim should be <= 5!")
        self._bounds = torch.tensor([[0.]*dim, [1.]*dim])
        super().__init__(noise_std=noise_std, negate=negate)

        self.max_timesteps = max_timesteps
        self.record_pos_every_n = record_pos_every_n
        self.outcome_dim = max_timesteps // record_pos_every_n

    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""
        Evaluate the objective function without noise for inputs in X.
        Args:
            X: `num_samples x self.dim` tensor of inputs
        Returns:
            trajectories: `num_samples x self.outcome_dim` tensor
        """
        trajectories = []
        save_stdout = sys.stdout
        sys.stdout = io.StringIO()  # suppress print out

        X_ = self._unstandardize_X(X, bounds = self.original_bounds[:, :self.dim].clone().detach())
        print(X, X_)

        for i, X_i in enumerate(X_):
            kwargs = {self.param_names[j]: p for j, p in enumerate(X_i)}
            # list of (x,y,z) tuples, length = max_timesteps // record_pos_every_n
            trajectory = spot_mini_mini_trajectory(
                max_timesteps=self.max_timesteps,
                record_pos_every_n=self.record_pos_every_n,
                **kwargs
            )
            # after flattening it should look like [x1,x2,...,y1,y2,...,z1,z2,...]
            flat_single_trajectory = torch.transpose(
                torch.tensor(trajectory, dtype=X.dtype), -2, -1).flatten()
            trajectories.append(flat_single_trajectory)

        sys.stdout = save_stdout  # restore print out

        trajectories = torch.stack(trajectories)

        return trajectories

    def _unstandardize_X(self, X, bounds):
            
        r"""
        Unstandardize the input X in [0,1] to the original bounds.

        Args:
            X: `num_samples x dim` tensor of inputs
            bounds: `2 x dim` tensor of bounds

        Returns:
            X: `num_samples x dim` tensor of unstandardized inputs
        """
        return X * (bounds[1] - bounds[0]) + bounds[0]