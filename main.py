import time
import sys
import os
import argparse

import numpy as np

from unitree_controller import UnitreeController


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--network", type=str, default="lo0")
    args = parser.parse_args()

    print("[WARNING]: Please ensure there are no obstacles around the robot while running this example.")
    input("[INPUT]: Press Enter to continue...")

    controller = UnitreeController(real=args.real, network_interface=args.network)
    controller.Init()

    try:
        controller.Start()
        print("[INFO]: Started")

        controller.Damp()
        time.sleep(1)

        controller.LockStanding()
        time.sleep(3)

        input("[INPUT]: Press Enter to start walking...")
        controller.StartMoving()
        time.sleep(1)
        controller.Move(0.5, 0.0, 0.0)

        time.sleep(1)

        controller.Move(0.0, 0.0, 0.0)

        time.sleep(1)

        controller.Damp()

        # while True:        
        #     time.sleep(1)
    except KeyboardInterrupt:
        controller.Stop()
        print("[INFO]: Stopped")