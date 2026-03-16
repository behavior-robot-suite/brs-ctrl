from collections import OrderedDict
import threading
import time
from typing import Dict

from brs_ctrl.robot_interface import R1ProInterface
from brs_ctrl.robot_interface.grippers.galaxea_g1 import GalaxeaR1ProGripper
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import get_default_context


def center_crop_resize_to_res(img, resolution):
    import cv2

    target_height = resolution
    target_width = resolution

    h, w = img.shape[:2]

    # Find the shorter dimension to determine the square crop size
    crop_size = min(h, w)

    # Calculate center crop coordinates
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    end_y = start_y + crop_size
    end_x = start_x + crop_size

    # Crop the center square
    img_cropped = img[start_y:end_y, start_x:end_x]

    # Resize to target dimensions
    img_resized = cv2.resize(img_cropped, (target_width, target_height), cv2.INTER_AREA)

    return img_resized


class R1ProEnv(gym.Env):
    def __init__(self, control_freq: float = 100.0, img_resolution: int = 256):
        super().__init__()
        if not get_default_context().ok():
            rclpy.init(args=None)
        self.robot_interface = R1ProInterface(
            control_freq=control_freq,
            left_gripper=GalaxeaR1ProGripper(
                left_or_right="left",
                gripper_close_stroke=0.0,
                gripper_open_stroke=100.0,
            ),
            right_gripper=GalaxeaR1ProGripper(
                left_or_right="right",
                gripper_close_stroke=0.0,
                gripper_open_stroke=100.0,
            ),
        )
        # Start a background executor so subscriptions/timers run
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.robot_interface)
        self._spin = True
        self._thread = threading.Thread(target=self._spin_thread, daemon=True)
        self._thread.start()

        self.observation_space = gym.spaces.Dict(
            {
                f"video.left_wrist_view_centercrop_res{img_resolution}": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(img_resolution, img_resolution, 3),
                    dtype=np.uint8,
                ),
                f"video.right_wrist_view_centercrop_res{img_resolution}": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(img_resolution, img_resolution, 3),
                    dtype=np.uint8,
                ),
                f"video.ego_view_centercrop_res{img_resolution}": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(img_resolution, img_resolution, 3),
                    dtype=np.uint8,
                ),
                "state.left_arm_joints": gym.spaces.Box(
                    shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32
                ),
                "state.right_arm_joints": gym.spaces.Box(
                    shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32
                ),
                "state.left_gripper": gym.spaces.Box(
                    shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32
                ),
                "state.right_gripper": gym.spaces.Box(
                    shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "action.left_arm_joints": gym.spaces.Box(
                    shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32
                ),
                "action.right_arm_joints": gym.spaces.Box(
                    shape=(7,), low=-np.inf, high=np.inf, dtype=np.float32
                ),
                "action.left_gripper": gym.spaces.Box(
                    shape=(1,), low=0, high=1, dtype=np.float32
                ),
                "action.right_gripper": gym.spaces.Box(
                    shape=(1,), low=0, high=1, dtype=np.float32
                ),
            }
        )
        self.img_resolution = img_resolution

        while True:
            last_rgb = self.robot_interface.last_rgb
            last_proprio = self.robot_interface.last_joint_position
            last_gripper = self.robot_interface.last_gripper_state
            if (
                last_rgb is not None
                and last_proprio is not None
                and last_gripper is not None
            ):
                break
            else:
                print(
                    f"Waiting for {last_rgb is None=}, {last_proprio is None=}, {last_gripper is None=} to be not None"  # noqa: E501
                )
                time.sleep(0.01)

    def _spin_thread(self):
        # run executor until close() flips the flag
        # Use shorter timeout for more responsive callback processing
        while self._spin and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.005)

    def step(self, action: Dict[str, np.ndarray]):
        # Process any pending callbacks to ensure fresh state before control
        # This ensures torso feedback is up-to-date for responsive control
        self._executor.spin_once(timeout_sec=0)

        self.robot_interface.control(
            arm_cmd={
                "left": action["action.left_arm_joints"],
                "right": action["action.right_arm_joints"],
            },
            gripper_cmd={
                "left": action["action.left_gripper"],
                "right": action["action.right_gripper"],
            },
            torso_cmd=action.get("torso", None),
            base_cmd=action.get("mobile_base", None),
        )
        return self._get_observation(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        # Process pending callbacks to ensure fresh state
        self._executor.spin_once(timeout_sec=0)
        obs = self._get_observation()
        return obs, {}

    def close(self):
        # Stop spinning and clean up the node
        self._spin = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        if hasattr(self, "_executor"):
            self._executor.remove_node(self.robot_interface)
        self.robot_interface.destroy_node()

        rclpy.shutdown()

    def _get_observation(self):
        last_rgb = self.robot_interface.last_rgb
        last_proprio = self.robot_interface.last_joint_position
        last_gripper = self.robot_interface.last_gripper_state
        obs_dict = OrderedDict()
        for k, v in last_rgb.items():
            name = k
            if k == "head":
                name = "ego"
            obs_dict[f"video.{name}_view_centercrop_res{self.img_resolution}"] = (
                center_crop_resize_to_res(v["img"], self.img_resolution)
            )

        # Split proprio into separate state components (removing torso)
        obs_dict["state.left_arm_joints"] = last_proprio["left_arm"]
        obs_dict["state.right_arm_joints"] = last_proprio["right_arm"]
        obs_dict["state.left_gripper"] = np.array(
            [last_gripper["left_gripper"]["gripper_position"]]
        )
        obs_dict["state.right_gripper"] = np.array(
            [last_gripper["right_gripper"]["gripper_position"]]
        )

        return obs_dict
