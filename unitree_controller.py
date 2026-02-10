import time
import sys
import os

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

import numpy as np

from move_sim import G1Config, InferenceEnv

G1_NUM_MOTOR = 29

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class UnitreeController:
    def __init__(self, real=False, network_interface="lo"):

        ChannelFactoryInitialize(0, network_interface)

        self.time_ = 0.0
        
        self.start_duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0

        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()
        self.real = real

        self.config = G1Config()
        self.control_dt_ = self.config.sim.dt

        self.motion_command = [0.0, 0.0, 0.0]

        if not real:
            print("[INFO]: Running in simulation mode.")
            self.update_mode_machine_ = True
        else:
            print("[INFO]: Running in real robot mode.")

    def Init(self):

        # if self.real:
        #     self.msc = MotionSwitcherClient()
        #     self.msc.SetTimeout(5.0)
        #     self.msc.Init()

        #     status, result = self.msc.CheckMode()
        #     while result['name']:
        #         self.msc.ReleaseMode()
        #         status, result = self.msc.CheckMode()
        #         time.sleep(1)

        # create publisher #
        if not self.real:
            self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.lowcmd_publisher_.Init()

            self.actions = np.zeros(12)
            self.env = InferenceEnv(self.config)
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()  

            self.motion_state = "damp"

        else:
            self.loco_client = LocoClient()
            self.loco_client.Init()

            self.lowcmd_publisher_ = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self.lowcmd_publisher_.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 1)

        time.sleep(1)  # wait for subscriber to be ready

        while self.low_state is None:
            print("[INFO]: Waiting robot connection...")
            time.sleep(1)
        print("[INFO]: Robot connected.")

    def Start(self):
        if not self.real:
            self.lowCmdWriteThreadPtr = RecurrentThread(
                interval=self.control_dt_, target=self.LowCmdWrite, name="control"
            )

            while self.update_mode_machine_ == False:
                time.sleep(1)

            if self.update_mode_machine_ == True:
                self.lowCmdWriteThreadPtr.Start()

    def Damp(self):

        if not self.real:
            self.motion_state = "damp"
        else:
            self.loco_client.Damp()

    def LockStanding(self):

        self.time_ = 0.0

        if not self.real:
            self.motion_state = "lock_standing"
        else:
            self.loco_client.LockStanding()


    def Move(self, vx: float, vy: float, vyaw: float):
        
        if not self.real:
            self.motion_command = [vx, vy, vyaw]
            self.motion_state = "walking"
        else:
            self.loco_client.Move(vx, vy, vyaw, continuous_move=True)
    

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
            print("Upated mode machine: ", self.mode_machine_)


    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        if self.motion_state == "lock_standing":
            # [Stage 1]: set robot to zero posture
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.start_duration_, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0. 
                self.low_cmd.motor_cmd[i].q = ratio * self.config.init_state.default_joint_angles[i] + \
                        (1 - ratio) * self.low_state.motor_state[i].q
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = self.config.control.Kp[i]
                self.low_cmd.motor_cmd[i].kd = self.config.control.Kd[i]            

        elif self.motion_state == "walking":
            # [Stage 3]: zero torque command
            self.low_cmd.mode_pr = Mode.PR
            self.low_cmd.mode_machine = self.mode_machine_


            if self.counter_ % self.config.sim.decimation == 0:
                self.actions = self.env.step(
                    low_state=self.low_state,
                    episode_timestep=self.counter_,
                    commands=np.array(self.motion_command, dtype=np.float32)
                )

            for i in range(15):
                self.low_cmd.motor_cmd[i].q = self.actions[i]
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = self.config.control.Kp[i]
                self.low_cmd.motor_cmd[i].kd = self.config.control.Kd[i]    
                self.low_cmd.motor_cmd[i].tau = 0. 
            for i in range(15, G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].q = self.config.init_state.default_joint_angles[i]
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = self.config.control.Kp[i]
                self.low_cmd.motor_cmd[i].kd = self.config.control.Kd[i]    
                self.low_cmd.motor_cmd[i].tau = 0.

            self.counter_ += 1

        elif self.motion_state == "damp":
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 0  # Disable all motors
                self.low_cmd.motor_cmd[i].tau = 0.
                self.low_cmd.motor_cmd[i].q = 0.
                self.low_cmd.motor_cmd[i].dq = 0.
                self.low_cmd.motor_cmd[i].kp = 0.
                self.low_cmd.motor_cmd[i].kd = 0.


        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)