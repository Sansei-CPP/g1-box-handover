import time
import sys
import os

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

import numpy as np
import torch
# import pypose as pp


class G1Config:
    
    phase_period = 0.8  # seconds

    class sim:
        dt =  0.005
        decimation = 5
        gravity = [0., 0. ,-9.81]  # [m/s^2]

    class init_state:
        pos = [0.0, 0.0, 0.8] # x, y, z for root link
    
        default_joint_angles_tensor = torch.tensor([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
            0.0, 0.0, 0.0,
            # 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.3,  0.25, 0.0, 0.97,  0.15, 0.0, 0.0,
            0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,
        ])  # for all 29
        default_joint_angles = default_joint_angles_tensor.numpy().tolist()

    class control:
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        Kp = [
            100, 100, 100, 150, 40, 40,      # legs
            100, 100, 100, 150, 40, 40,      # legs
            200, 40, 40,                   # waist
            40, 40, 40, 40, 40, 40, 40,  # arms
            40, 40, 40, 40, 40, 40, 40,  # arms
        ]

        Kd = [
            2, 2, 2, 4, 2, 2,     # legs
            2, 2, 2, 4, 2, 2,     # legs
            5, 5, 5,              # waist
            1, 1, 1, 1, 1, 1, 1,  # arms
            1, 1, 1, 1, 1, 1, 1,  # arms
        ]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        num_motor = 29

        sdk_to_isaac = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 4, 10, 5, 11]
        isaac_to_sdk = [0, 3, 6, 9, 11, 13, 1, 4, 7, 10, 12, 14, 2, 5, 8]

    class env:
        num_observations = 54
        history_length = 5
        num_actions = 15

    class policy:
        model_path = os.path.join('weights', 'policy_better.pt')

    class normalization:
        class obs_scales:
            ang_vel = 0.2
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        # clip_observations = 100.
        # clip_actions = 100.
        # cmd_scale = torch.tensor([2.0, 2.0, 0.25])
        # max_cmd = torch.tensor([0.8, 0.5, 1.57])
    


class InferenceEnv:
    def __init__(self, config: G1Config) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        
        self.config = config
        self.past_actions = torch.zeros(self.config.env.num_actions)
        self.observations_holder = torch.zeros(
            self.config.env.history_length, self.config.env.num_observations
        )

         # load policy
        self.policy = torch.jit.load(self.config.policy.model_path).to(self.device)
        self.policy.eval()

    def step(self, low_state: LowState_, episode_timestep: int, commands: np.ndarray = None):

        observations = self.compute_observations(
            low_state=low_state,
            episode_timestep=episode_timestep,
            commands=commands,
        )
        final_observation = self.process_observation_history(observations)

        with torch.no_grad():
            actions = self.policy(final_observation.unsqueeze(0).to(self.device)).squeeze(0)

        self.past_actions = actions.cpu() 

        return (
            actions.cpu()[self.config.control.isaac_to_sdk] * \
                self.config.control.action_scale + self.config.init_state.default_joint_angles_tensor[:15]).numpy()

    def process_observation_history(self, new_observation: torch.Tensor):
        L = self.config.env.history_length

        self.observations_holder = torch.roll(self.observations_holder, shifts=-1, dims=0)
        self.observations_holder[-1, :] = new_observation
       
        final_observation = torch.zeros(
            self.config.env.num_observations, L
        )
        
        final_observation[0:3, :] = self.observations_holder[:, 0:3].reshape(3, L)  # base ang vel
        final_observation[3:6, :] = self.observations_holder[:, 3:6].reshape(3, L)  # projected gravity
        final_observation[6:9, :] = self.observations_holder[:, 6:9].reshape(3, L)  # commands
        final_observation[9:24, :] = self.observations_holder[:, 9:24].reshape(15, L)  # dof pos
        final_observation[24:39, :] = self.observations_holder[:, 24:39].reshape(15, L)  # dof vel
        final_observation[39:54, :] = self.observations_holder[:, 39:54].reshape(15, L)  # past actions

        return final_observation.reshape(-1)

    def compute_observations(
        self,
        low_state: LowState_,
        episode_timestep: int,
        commands: np.ndarray = None,
    ):
        
        base_ang_vel = torch.tensor(low_state.imu_state.gyroscope) * self.config.normalization.obs_scales.ang_vel # 3

        projected_gravity = self.get_gravity_orientation(low_state.imu_state.quaternion)  # 3

        joy_state = self.read_joy_state(low_state.wireless_remote)
        if commands is None:
            commands = torch.tensor(
                [joy_state['ly'], -joy_state['lx'], -joy_state['rx']]
            ) # 3
        else:
            commands = torch.tensor(commands)  # 3
        # commands = torch.zeros(3)  # 3

        dof_pos = torch.zeros(15) # 15
        dof_vel = torch.zeros(15) # 15
        for i in range(15):
            isaac_idx = self.config.control.sdk_to_isaac[i]
            dof_pos[i] = (low_state.motor_state[isaac_idx].q - self.config.init_state.default_joint_angles_tensor[isaac_idx]) * self.config.normalization.obs_scales.dof_pos
            dof_vel[i] = low_state.motor_state[isaac_idx].dq * self.config.normalization.obs_scales.dof_vel

        actions = self.past_actions  # 15

        observations = torch.cat([
            base_ang_vel, # 3
            projected_gravity, # 3
            commands, # 3
            dof_pos, # 15
            dof_vel, # 15
            actions, # 15
        ], dim=0)  # 54

        return observations


    def read_joy_state(self, joy_data):
        
        lx = np.array([joy_data[4], joy_data[5], joy_data[6], joy_data[7]], dtype=np.uint8).view(np.float32)[0]
        rx = np.array([joy_data[8], joy_data[9], joy_data[10], joy_data[11]], dtype=np.uint8).view(np.float32)[0]
        ry = np.array([joy_data[12], joy_data[13], joy_data[14], joy_data[15]], dtype=np.uint8).view(np.float32)[0]
        ly = np.array([joy_data[20], joy_data[21], joy_data[22], joy_data[23]], dtype=np.uint8).view(np.float32)[0]


        return {'lx': lx, 'ly': ly, 'rx': rx, 'ry': ry}
    
    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = torch.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
    

joints = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

joint_id_map = {name: idx for idx, name in enumerate(joints)}

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof
