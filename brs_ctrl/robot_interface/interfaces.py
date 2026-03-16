from typing import Literal, Optional, Dict, Union
from functools import partial
import time

# ROS1 related imports for R1Interface
try:
    import rospy
    import ros_numpy
except ImportError as e:
    print(f"Failed to import ROS related modules, R1Interface won't work.")
    print(e)

# ROS2 related imports for R1ProInterface
try:
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from builtin_interfaces.msg import Time as TimeMsg
except ImportError as e:
    print(f"Failed to import ROS2 related modules, R1ProInterface won't work.")
    print(e)

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import JointState, PointCloud2, Image, CompressedImage
import numpy as np
from cv_bridge import CvBridge

import brs_ctrl.utils as U
from brs_ctrl.kinematics import R1Kinematics
from brs_ctrl.robot_interface.grippers.base import R1BaseGripper, R1ProBaseGripper
from brs_ctrl.robot_interface.utils import get_xyz_points
from brs_ctrl.robot_interface.mobile_base import Odom


class R1Interface:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])

    def __init__(
        self,
        *,
        # ====== left arm ======
        left_arm_joint_state_topic: str = "/hdas/feedback_arm_left",
        left_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_left",
        left_gripper: Optional[R1BaseGripper] = None,
        # ====== right arm ======
        right_arm_joint_state_topic: str = "/hdas/feedback_arm_right",
        right_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_right",
        right_gripper: Optional[R1BaseGripper] = None,
        # ====== torso ======
        torso_joint_state_topic: str = "/hdas/feedback_torso",
        torso_joint_target_position_topic: str = "/motion_target/target_joint_state_torso",
        # ====== mobile base ======
        odometry_topic: str = "/camera/odom/sample",
        wait_for_first_odom_msg: bool = False,
        mobile_base_vel_cmd_topic: str = "/motion_target/target_speed_chassis",
        mobile_base_cmd_threshold: Union[np.ndarray, float] = np.array(
            [0.01, 0.01, 0.05]
        ),
        mobile_base_cmd_limit: Union[np.ndarray, float] = np.array([0.3, 0.3, 0.4]),
        # ====== cameras ======
        enable_pointcloud: bool = False,
        pointcloud_topic: str = "/r1_jetson/fused_pcd",
        enable_rgbd: bool = False,
        rgb_topics: Optional[Dict[str, str]] = None,
        depth_topics: Optional[Dict[str, str]] = None,
        # ====== common ======
        publisher_node_name: str = "joint_state_publisher",
        publisher_node_queue_size: int = 10,
        state_buffer_size: int = 1000,
        arm_joint_control_step_interval: float = 0.4,
        torso_joint_control_step_interval: float = 0.1,
        control_freq: float = 100.0,
        on_arm_cmd_out_of_range: Literal["raise", "clip"] = "clip",
        on_torso_cmd_out_of_range: Literal["raise", "clip"] = "clip",
    ):
        self._kin_model = R1Kinematics()

        self._left_arm_joint_state_buffer = None
        self._right_arm_joint_state_buffer = None
        self._torso_joint_state_buffer = None
        self._state_buffer_size = state_buffer_size
        self._pcd = None
        self._rgb = None
        self._depth = None

        # ros node initialization
        rospy.init_node(publisher_node_name, anonymous=True)
        self._rate = rospy.Rate(control_freq)

        # command publishers
        self._left_arm_joint_target_position_pub = rospy.Publisher(
            left_arm_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )
        self._right_arm_joint_target_position_pub = rospy.Publisher(
            right_arm_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )
        self._torso_joint_target_position_pub = rospy.Publisher(
            torso_joint_target_position_topic,
            JointState,
            queue_size=publisher_node_queue_size,
        )

        # sensor subscribers
        self._left_arm_joint_state_sub = rospy.Subscriber(
            left_arm_joint_state_topic,
            JointState,
            self._left_arm_state_callback,
        )
        self._right_arm_joint_state_sub = rospy.Subscriber(
            right_arm_joint_state_topic,
            JointState,
            self._right_arm_state_callback,
        )
        self._torso_joint_state_sub = rospy.Subscriber(
            torso_joint_state_topic,
            JointState,
            self._torso_state_callback,
        )

        if enable_pointcloud:
            self._pcd_sub = rospy.Subscriber(
                pointcloud_topic,
                PointCloud2,
                self._pcd_callback,
            )
        if enable_rgbd:
            rgb_topics = rgb_topics or {
                "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/rgb/image_rect_color",
                "left_wrist": "/zed_multi_cams/zed2_left_wrist/zed_nodelet_left_wrist/rgb/image_rect_color",
                "right_wrist": "/zed_multi_cams/zed2_right_wrist/zed_nodelet_right_wrist/rgb/image_rect_color",
            }
            depth_topics = depth_topics or {
                "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/depth/depth_registered",
                "left_wrist": "/zed_multi_cams/zed2_left_wrist/zed_nodelet_left_wrist/depth/depth_registered",
                "right_wrist": "/zed_multi_cams/zed2_right_wrist/zed_nodelet_right_wrist/depth/depth_registered",
            }
            self._rgb = {k: None for k in rgb_topics}
            self._depth = {k: None for k in depth_topics}
            self._cv_bridge = CvBridge()
            self._rgb_subs = {
                k: rospy.Subscriber(v, Image, partial(self._rgb_callback, name=k))
                for k, v in rgb_topics.items()
            }
            self._depth_subs = {
                k: rospy.Subscriber(v, Image, partial(self._depth_callback, name=k))
                for k, v in depth_topics.items()
            }
        else:
            self._cv_bridge = self._rgb_subs = self._depth_subs = None

        # mobile base
        self._odom = Odom(
            odom_topic=odometry_topic,
            T_odom2base=self._kin_model.T_odom2base,
            wait_for_first_msg=wait_for_first_odom_msg,
        )
        if isinstance(mobile_base_cmd_threshold, float):
            mobile_base_cmd_threshold = np.array(
                [
                    mobile_base_cmd_threshold,
                ]
                * 3
            )
        if isinstance(mobile_base_cmd_limit, float):
            mobile_base_cmd_limit = np.array(
                [
                    mobile_base_cmd_limit,
                ]
                * 3
            )
        assert mobile_base_cmd_threshold.shape == mobile_base_cmd_limit.shape == (3,)
        self._mobile_base_cmd_threshold = mobile_base_cmd_threshold
        self._mobile_base_cmd_limit = mobile_base_cmd_limit
        self._mobile_base_vel_cmd_pub = rospy.Publisher(
            mobile_base_vel_cmd_topic, Twist, queue_size=publisher_node_queue_size
        )

        self._left_gripper, self._right_gripper = left_gripper, right_gripper
        if self._left_gripper is not None:
            self._left_gripper.init_hook()
        if self._right_gripper is not None:
            self._right_gripper.init_hook()

        assert on_arm_cmd_out_of_range in ["raise", "clip"]
        self._on_arm_cmd_out_of_range = on_arm_cmd_out_of_range
        assert on_torso_cmd_out_of_range in ["raise", "clip"]
        self._on_torso_cmd_out_of_range = on_torso_cmd_out_of_range

        self._arm_joint_control_step_interval = arm_joint_control_step_interval
        self._torso_joint_control_step_interval = torso_joint_control_step_interval

    def control(
        self,
        *,
        arm_controller: Literal["joint_position"] = "joint_position",
        arm_cmd: Dict[str, Optional[np.ndarray]],
        gripper_cmd: Optional[Dict[str, float]] = None,
        torso_controller: Literal["joint_position"] = "joint_position",
        torso_cmd: Optional[np.ndarray] = None,
        base_cmd: Optional[np.ndarray] = None,
    ):
        assert (
            arm_controller == "joint_position"
        ), f"Invalid arm controller {arm_controller}"
        assert (
            torso_controller == "joint_position"
        ), f"Invalid torso controller {torso_controller}"
        assert "left" in arm_cmd and "right" in arm_cmd
        if gripper_cmd is not None:
            assert "left" in gripper_cmd and "right" in gripper_cmd

        # base control
        base_cmd = np.zeros((3,)) if base_cmd is None else base_cmd
        try:
            self._mobile_base_control(base_cmd)
        except rospy.ROSInterruptException:
            pass

        # upper-body control
        try:
            self._upper_body_joint_position_control(
                left_arm_target_q=arm_cmd["left"],
                right_arm_target_q=arm_cmd["right"],
                torso_target_q=torso_cmd,
            )
        except rospy.ROSInterruptException:
            pass

        # optional gripper control
        if gripper_cmd is not None:
            try:
                self._gripper_control(gripper_cmd)
            except rospy.ROSInterruptException:
                pass

    def _mobile_base_control(self, cmd: np.ndarray):
        set_zero = np.abs(cmd) < self._mobile_base_cmd_threshold
        cmd[set_zero] = 0
        cmd = np.clip(cmd, -self._mobile_base_cmd_limit, self._mobile_base_cmd_limit)
        _cmd = Twist()
        _cmd.linear.x = cmd[0]
        _cmd.linear.y = cmd[1]
        _cmd.angular.z = cmd[2]
        self._mobile_base_vel_cmd_pub.publish(_cmd)

    def stop_mobile_base(self):
        self._mobile_base_vel_cmd_pub.publish(Twist())

    def _upper_body_joint_position_control(
        self,
        left_arm_target_q: Optional[np.ndarray] = None,
        right_arm_target_q: Optional[np.ndarray] = None,
        torso_target_q: Optional[np.ndarray] = None,
    ):
        # shape check
        if left_arm_target_q is not None:
            assert left_arm_target_q.shape == (
                6,
            ), f"Expected left_arm_target_q to have shape (6,), got {left_arm_target_q.shape}"
        if right_arm_target_q is not None:
            assert right_arm_target_q.shape == (
                6,
            ), f"Expected right_arm_target_q to have shape (6,), got {right_arm_target_q.shape}"
        if torso_target_q is not None:
            assert torso_target_q.shape == (
                4,
            ), f"Expected torso_target_q to have shape (4,), got {torso_target_q.shape}"

        # value out-of-range check
        if left_arm_target_q is not None:
            left_in_range = np.logical_and(
                left_arm_target_q >= self.left_arm_joint_low,
                left_arm_target_q <= self.left_arm_joint_high,
            )
            left_invalid_joint_idxs = np.where(~left_in_range)[0]
            if len(left_invalid_joint_idxs) > 0:
                for idx in left_invalid_joint_idxs:
                    msg = (
                        f"Left arm joint {idx + 1} target position {left_arm_target_q[idx]} is out of range "
                        f"[{self.left_arm_joint_low[idx]}, {self.left_arm_joint_high[idx]}]."
                    )
                    if self._on_arm_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        left_arm_target_q[idx] = np.clip(
                            left_arm_target_q[idx],
                            self.left_arm_joint_low[idx],
                            self.left_arm_joint_high[idx],
                        )
                    else:
                        raise ValueError(msg)
                    rospy.logwarn(msg)

        if right_arm_target_q is not None:
            right_in_range = np.logical_and(
                right_arm_target_q >= self.right_arm_joint_low,
                right_arm_target_q <= self.right_arm_joint_high,
            )
            right_invalid_joint_idxs = np.where(~right_in_range)[0]
            if len(right_invalid_joint_idxs) > 0:
                for idx in right_invalid_joint_idxs:
                    msg = (
                        f"Right arm joint {idx + 1} target position {right_arm_target_q[idx]} is out of range "
                        f"[{self.right_arm_joint_low[idx]}, {self.right_arm_joint_high[idx]}]."
                    )
                    if self._on_arm_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        right_arm_target_q[idx] = np.clip(
                            right_arm_target_q[idx],
                            self.right_arm_joint_low[idx],
                            self.right_arm_joint_high[idx],
                        )
                    else:
                        raise ValueError(msg)
                    rospy.logwarn(msg)

        if torso_target_q is not None:
            torso_in_range = np.logical_and(
                torso_target_q >= self.torso_joint_low,
                torso_target_q <= self.torso_joint_high,
            )
            torso_invalid_joint_idxs = np.where(~torso_in_range)[0]
            if len(torso_invalid_joint_idxs) > 0:
                for idx in torso_invalid_joint_idxs:
                    msg = (
                        f"Torso joint {idx + 1} target position {torso_target_q[idx]} is out of range "
                        f"[{self.torso_joint_low[idx]}, {self.torso_joint_high[idx]}]."
                    )
                    if self._on_torso_cmd_out_of_range == "clip":
                        msg += " Clipping to range."
                        torso_target_q[idx] = np.clip(
                            torso_target_q[idx],
                            self.torso_joint_low[idx],
                            self.torso_joint_high[idx],
                        )
                        rospy.logwarn(msg)
                    else:
                        raise ValueError(msg)

        # compute # steps for cmds
        last_two_arms_joint_positions = self.last_joint_position
        left_arm_joint_state = JointState()
        left_arm_joint_state.position = last_two_arms_joint_positions["left_arm"]
        left_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                left_arm_target_q
                                - np.array(left_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if left_arm_target_q is not None
            else 0
        )
        right_arm_joint_state = JointState()
        right_arm_joint_state.position = last_two_arms_joint_positions["right_arm"]
        right_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                right_arm_target_q
                                - np.array(right_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if right_arm_target_q is not None
            else 0
        )
        last_torso_joint_positions = self.last_joint_position["torso"]
        torso_joint_state = JointState()
        torso_joint_state.position = last_torso_joint_positions
        torso_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (torso_target_q - np.array(torso_joint_state.position))
                            / self._torso_joint_control_step_interval
                        )
                    )
                )
            )
            if torso_target_q is not None
            else 0
        )

        # TORSO: Publish immediately without interpolation for responsive control
        # This ensures holding the torso button results in continuous movement
        if torso_target_q is not None:
            torso_joint_state.header.stamp = rospy.Time.now()
            torso_joint_state.position = torso_target_q.tolist()
            self._torso_joint_target_position_pub.publish(torso_joint_state)

        # ARMS: Use interpolation loop only for arms (torso already published above)
        max_arm_n_steps = max(left_arm_n_steps, right_arm_n_steps)
        if max_arm_n_steps > 1:
            if left_arm_target_q is not None:
                left_arm_increment = (
                    (left_arm_target_q - np.array(left_arm_joint_state.position))
                    / (left_arm_n_steps - 1)
                    if left_arm_n_steps > 1
                    else (left_arm_target_q - np.array(left_arm_joint_state.position))
                )
            else:
                left_arm_increment = np.zeros(6)
            if right_arm_target_q is not None:
                right_arm_increment = (
                    (right_arm_target_q - np.array(right_arm_joint_state.position))
                    / (right_arm_n_steps - 1)
                    if right_arm_n_steps > 1
                    else (right_arm_target_q - np.array(right_arm_joint_state.position))
                )
            else:
                right_arm_increment = np.zeros(6)

            for step in range(max_arm_n_steps - 1):
                if left_arm_target_q is not None:
                    left_arm_joint_state.header.stamp = rospy.Time.now()
                    if step <= left_arm_n_steps - 1:
                        left_arm_joint_state.position = (
                            np.array(left_arm_joint_state.position) + left_arm_increment
                        ).tolist()
                    else:
                        left_arm_joint_state.position = left_arm_target_q.tolist()
                    self._left_arm_joint_target_position_pub.publish(
                        left_arm_joint_state
                    )
                if right_arm_target_q is not None:
                    right_arm_joint_state.header.stamp = rospy.Time.now()
                    if step <= right_arm_n_steps - 1:
                        right_arm_joint_state.position = (
                            np.array(right_arm_joint_state.position)
                            + right_arm_increment
                        ).tolist()
                    else:
                        right_arm_joint_state.position = right_arm_target_q.tolist()
                    self._right_arm_joint_target_position_pub.publish(
                        right_arm_joint_state
                    )
                self._rate.sleep()

        # ensure exact final target positions for arms
        if left_arm_target_q is not None:
            left_arm_joint_state.header.stamp = rospy.Time.now()
            left_arm_joint_state.position = left_arm_target_q.tolist()
            self._left_arm_joint_target_position_pub.publish(left_arm_joint_state)
        if right_arm_target_q is not None:
            right_arm_joint_state.header.stamp = rospy.Time.now()
            right_arm_joint_state.position = right_arm_target_q.tolist()
            self._right_arm_joint_target_position_pub.publish(right_arm_joint_state)
        if not (
            left_arm_target_q is None
            and right_arm_target_q is None
            and torso_target_q is None
        ):
            self._rate.sleep()

    def _gripper_control(self, gripper_action: Dict[str, float]):
        left_gripper_action = gripper_action["left"]
        right_gripper_action = gripper_action["right"]
        assert (
            1 >= left_gripper_action >= 0
        ), "Invalid left gripper action, must be between 0 and 1"
        assert (
            1 >= right_gripper_action >= 0
        ), "Invalid right gripper action, must be between 0 and 1"

        if self._left_gripper is not None:
            self._left_gripper.act(left_gripper_action)
        if self._right_gripper is not None:
            self._right_gripper.act(right_gripper_action)

    def _pcd_callback(self, pcd: PointCloud2):
        timestamp = pcd.header.stamp.secs + pcd.header.stamp.nsecs * 1e-9
        pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pcd)
        pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        self._pcd = {
            "xyz": pcd_xyz,
            "rgb": pcd_rgb,
            "stamp": timestamp,
        }

    def _rgb_callback(self, rgb_msg: Image, name: str):
        timestamp = rgb_msg.header.stamp.secs + rgb_msg.header.stamp.nsecs * 1e-9
        img = np.asarray(self._cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8"))
        self._rgb[name] = {
            "img": img,
            "stamp": timestamp,
        }

    def _depth_callback(self, depth_msg: Image, name: str):
        timestamp = depth_msg.header.stamp.secs + depth_msg.header.stamp.nsecs * 1e-9
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        self._depth[name] = {
            "depth": depth_image,
            "stamp": timestamp,
        }

    def _left_arm_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:6]]),
                self.left_arm_joint_low,
                self.left_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:6]),
            "joint_effort": np.array([data.effort][:6]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._left_arm_joint_state_buffer is None:
            self._left_arm_joint_state_buffer = new_state
        else:
            self._left_arm_joint_state_buffer = U.any_concat(
                [
                    self._left_arm_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._left_arm_joint_state_buffer = U.any_slice(
                self._left_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _right_arm_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:6]]),
                self.right_arm_joint_low,
                self.right_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:6]),
            "joint_effort": np.array([data.effort][:6]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._right_arm_joint_state_buffer is None:
            self._right_arm_joint_state_buffer = new_state
        else:
            self._right_arm_joint_state_buffer = U.any_concat(
                [
                    self._right_arm_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._right_arm_joint_state_buffer = U.any_slice(
                self._right_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _torso_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:4]]),
                self.torso_joint_low,
                self.torso_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:4]),
            "joint_effort": np.array([data.effort][:4]),
            "seq": np.array([data.header.seq]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }

        if self._torso_joint_state_buffer is None:
            self._torso_joint_state_buffer = new_state
        else:
            self._torso_joint_state_buffer = U.any_concat(
                [
                    self._torso_joint_state_buffer,
                    new_state,
                ],
                dim=0,
            )
            self._torso_joint_state_buffer = U.any_slice(
                self._torso_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def close(self):
        # stop the base
        self.stop_mobile_base()
        if self._left_gripper is not None:
            self._left_gripper.close()
        if self._right_gripper is not None:
            self._right_gripper.close()
        rospy.signal_shutdown("Shutting down node")

    def __del__(self):
        self.close()

    @property
    def last_joint_position(self) -> Dict[str, np.ndarray]:
        return {
            "left_arm": U.any_slice(self._left_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "right_arm": U.any_slice(self._right_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "torso": U.any_slice(self._torso_joint_state_buffer, -1)["joint_position"],
        }

    @property
    def last_gripper_state(self):
        return {
            "left_gripper": (
                None
                if self._left_gripper is None
                else U.any_slice(self._left_gripper.state_buffer, -1)
            ),
            "right_gripper": (
                None
                if self._right_gripper is None
                else U.any_slice(self._right_gripper.state_buffer, -1)
            ),
        }

    @property
    def last_pointcloud(self):
        return {
            "xyz": self._pcd["xyz"],
            "rgb": self._pcd["rgb"],
            "stamp": self._pcd["stamp"],
        }

    @property
    def last_rgb(self):
        return (
            {
                k: {
                    "img": v["img"],
                    "stamp": v["stamp"],
                }
                for k, v in self._rgb.items()
            }
            if self._rgb is not None
            else None
        )

    @property
    def last_depth(self):
        return (
            {
                k: {
                    "depth": v["depth"],
                    "stamp": v["stamp"],
                }
                for k, v in self._depth.items()
            }
            if self._depth is not None
            else None
        )

    @property
    def curr_base_pose(self):
        return self._odom.curr_base_pose

    @property
    def curr_base_position(self):
        return self._odom.curr_base_position

    @property
    def curr_base_orientation(self):
        return self._odom.curr_base_orientation

    @property
    def curr_base_velocity(self):
        return self._odom.curr_base_velocity


class R1ProInterface(Node):
    torso_joint_high = np.array([1.8326, 2.5307, 1.5708, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -1.8326, -3.0543])
    left_arm_joint_high = np.array(
        [1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708]
    )
    left_arm_joint_low = np.array(
        [-4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708]
    )
    right_arm_joint_high = np.array(
        [1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708]
    )
    right_arm_joint_low = np.array(
        [-4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708]
    )

    def __init__(
        self,
        *,
        # ====== left arm ======
        left_arm_joint_state_topic: str = "/hdas/feedback_arm_left",
        left_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_left",
        left_gripper: Optional[R1ProBaseGripper] = None,
        # ====== right arm ======
        right_arm_joint_state_topic: str = "/hdas/feedback_arm_right",
        right_arm_joint_target_position_topic: str = "/motion_target/target_joint_state_arm_right",
        right_gripper: Optional[R1ProBaseGripper] = None,
        # ====== torso ======
        torso_joint_state_topic: str = "/hdas/feedback_torso",
        torso_joint_target_position_topic: str = "/motion_target/target_joint_state_torso",
        # ====== chassis ======
        chassis_joint_state_topic: str = "/hdas/feedback_chassis",
        # ====== mobile base ======
        mobile_base_vel_cmd_topic: str = "/motion_target/target_speed_chassis",
        mobile_base_cmd_threshold: Union[np.ndarray, float] = np.array(
            [0.01, 0.01, 0.05]
        ),
        mobile_base_cmd_limit: Union[np.ndarray, float] = np.array([0.3, 0.3, 0.4]),
        # ====== odometry ======
        odometry_topic: str = "/camera/odom/sample",
        T_odom2base: Optional[np.ndarray] = None,
        wait_for_first_odom_msg: bool = False,
        # ====== cameras ======
        enable_rgb: bool = True,
        rgb_topics: Optional[Dict[str, str]] = None,
        # ====== common ======
        publisher_node_name: str = "joint_state_publisher",
        publisher_node_queue_size: int = 10,
        state_buffer_size: int = 1000,
        arm_joint_control_step_interval: float = 0.4,
        torso_joint_control_step_interval: float = 0.1,
        control_freq: float = 100.0,
        on_arm_cmd_out_of_range: Literal["raise", "clip"] = "clip",
        on_torso_cmd_out_of_range: Literal["raise", "clip"] = "clip",
        log_clipping_warnings: bool = True,
    ):
        super().__init__(publisher_node_name)
        # Frequency for sleeps (wall clock; prefer timers for periodic callbacks)
        self._control_freq = float(control_freq)
        self._sleep_dt = 1.0 / self._control_freq
        self._log_clipping_warnings = log_clipping_warnings

        self._left_arm_joint_state_buffer = None
        self._right_arm_joint_state_buffer = None
        self._torso_joint_state_buffer = None
        self._chassis_joint_state_buffer = None
        self._state_buffer_size = state_buffer_size
        self._rgb = None

        # QoS
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=publisher_node_queue_size,
            durability=DurabilityPolicy.VOLATILE,
        )
        sens_qos = qos_profile_sensor_data

        # command publishers
        self._left_arm_joint_target_position_pub = self.create_publisher(
            JointState, left_arm_joint_target_position_topic, cmd_qos
        )
        self._right_arm_joint_target_position_pub = self.create_publisher(
            JointState, right_arm_joint_target_position_topic, cmd_qos
        )
        self._torso_joint_target_position_pub = self.create_publisher(
            JointState, torso_joint_target_position_topic, cmd_qos
        )

        # sensor subscribers
        self._left_arm_joint_state_sub = self.create_subscription(
            JointState,
            left_arm_joint_state_topic,
            self._left_arm_state_callback,
            sens_qos,
        )
        self._right_arm_joint_state_sub = self.create_subscription(
            JointState,
            right_arm_joint_state_topic,
            self._right_arm_state_callback,
            sens_qos,
        )
        self._torso_joint_state_sub = self.create_subscription(
            JointState, torso_joint_state_topic, self._torso_state_callback, sens_qos
        )
        self._chassis_joint_state_sub = self.create_subscription(
            JointState, chassis_joint_state_topic, self._chassis_state_callback, sens_qos
        )

        if enable_rgb:
            rgb_topics = rgb_topics or {
                "head": "/hdas/camera_head/left_raw/image_raw_color/compressed",
                "left_wrist": "/hdas/camera_wrist_left/color/image_rect_raw/compressed",
                "right_wrist": "/hdas/camera_wrist_right/color/image_rect_raw/compressed",
            }
            self._rgb = {k: None for k in rgb_topics}
            self._cv_bridge = CvBridge()
            self._rgb_subs = {
                k: self.create_subscription(
                    CompressedImage, v, partial(self._rgb_callback, name=k), sens_qos
                )
                for k, v in rgb_topics.items()
            }
        else:
            self._cv_bridge = self._rgb_subs = None

        # mobile base
        if isinstance(mobile_base_cmd_threshold, float):
            mobile_base_cmd_threshold = np.array(
                [
                    mobile_base_cmd_threshold,
                ]
                * 3
            )
        if isinstance(mobile_base_cmd_limit, float):
            mobile_base_cmd_limit = np.array(
                [
                    mobile_base_cmd_limit,
                ]
                * 3
            )
        assert mobile_base_cmd_threshold.shape == mobile_base_cmd_limit.shape == (3,)
        self._mobile_base_cmd_threshold = mobile_base_cmd_threshold
        self._mobile_base_cmd_limit = mobile_base_cmd_limit
        self._mobile_base_vel_cmd_pub = self.create_publisher(
            TwistStamped, mobile_base_vel_cmd_topic, cmd_qos
        )

        # odometry
        if T_odom2base is None:
            T_odom2base = np.eye(4)  # Identity if no transform provided
        self._odom = Odom(
            self,  # ROS2 node
            odom_topic=odometry_topic,
            T_odom2base=T_odom2base,
            wait_for_first_msg=wait_for_first_odom_msg,
        )

        self._left_gripper, self._right_gripper = left_gripper, right_gripper
        if self._left_gripper is not None:
            self._left_gripper.init_hook(self)
        if self._right_gripper is not None:
            self._right_gripper.init_hook(self)

        assert on_arm_cmd_out_of_range in ["raise", "clip"]
        self._on_arm_cmd_out_of_range = on_arm_cmd_out_of_range
        assert on_torso_cmd_out_of_range in ["raise", "clip"]
        self._on_torso_cmd_out_of_range = on_torso_cmd_out_of_range

        self._arm_joint_control_step_interval = arm_joint_control_step_interval
        self._torso_joint_control_step_interval = torso_joint_control_step_interval

    def control(
        self,
        *,
        arm_controller: Literal["joint_position"] = "joint_position",
        arm_cmd: Dict[str, Optional[np.ndarray]],
        gripper_cmd: Optional[Dict[str, float]] = None,
        torso_controller: Literal["joint_position"] = "joint_position",
        torso_cmd: Optional[np.ndarray] = None,
        base_cmd: Optional[np.ndarray] = None,
    ):
        assert (
            arm_controller == "joint_position"
        ), f"Invalid arm controller {arm_controller}"
        assert (
            torso_controller == "joint_position"
        ), f"Invalid torso controller {torso_controller}"
        assert "left" in arm_cmd and "right" in arm_cmd
        if gripper_cmd is not None:
            assert "left" in gripper_cmd and "right" in gripper_cmd

        # base control
        base_cmd = np.zeros((3,)) if base_cmd is None else base_cmd
        try:
            self._mobile_base_control(base_cmd)
        except Exception as e:
            self.get_logger().warning(f"Base control interrupted: {e}")

        # upper-body control
        try:
            self._upper_body_joint_position_control(
                left_arm_target_q=arm_cmd["left"],
                right_arm_target_q=arm_cmd["right"],
                torso_target_q=torso_cmd,
            )
        except Exception as e:
            self.get_logger().warning(f"Upper-body control interrupted: {e}")

        # optional gripper control
        if gripper_cmd is not None:
            try:
                self._gripper_control(gripper_cmd)
            except Exception as e:
                self.get_logger().warning(f"Gripper control interrupted: {e}")

    # ---------- Helpers ----------
    def _now_msg(self) -> TimeMsg:
        return self.get_clock().now().to_msg()

    def _sleep_tick(self):
        time.sleep(self._sleep_dt)

    def _mobile_base_control(self, cmd: np.ndarray):
        cmd = cmd.copy()
        set_zero = np.abs(cmd) < self._mobile_base_cmd_threshold
        cmd[set_zero] = 0.0
        cmd = np.clip(cmd, -self._mobile_base_cmd_limit, self._mobile_base_cmd_limit)
        msg = TwistStamped()
        twist_msg = Twist()
        twist_msg.linear.x = float(cmd[0])
        twist_msg.linear.y = float(cmd[1])
        twist_msg.angular.z = float(cmd[2])
        msg.twist = twist_msg
        self._mobile_base_vel_cmd_pub.publish(msg)

    def stop_mobile_base(self):
        msg = TwistStamped()
        self._mobile_base_vel_cmd_pub.publish(msg)

    def _upper_body_joint_position_control(
        self,
        left_arm_target_q: Optional[np.ndarray] = None,
        right_arm_target_q: Optional[np.ndarray] = None,
        torso_target_q: Optional[np.ndarray] = None,
    ):
        # shape checks
        if left_arm_target_q is not None:
            assert left_arm_target_q.shape == (
                7,
            ), f"Expected (7,), got {left_arm_target_q.shape}"
        if right_arm_target_q is not None:
            assert right_arm_target_q.shape == (
                7,
            ), f"Expected (7,), got {right_arm_target_q.shape}"
        if torso_target_q is not None:
            assert torso_target_q.shape == (
                4,
            ), f"Expected (4,), got {torso_target_q.shape}"

        # range checks (with clipping if chosen)
        if left_arm_target_q is not None:
            left_in_range = np.logical_and(
                left_arm_target_q >= self.left_arm_joint_low,
                left_arm_target_q <= self.left_arm_joint_high,
            )
            for idx in np.where(~left_in_range)[0]:
                if self._on_arm_cmd_out_of_range == "clip":
                    left_arm_target_q[idx] = np.clip(
                        left_arm_target_q[idx],
                        self.left_arm_joint_low[idx],
                        self.left_arm_joint_high[idx],
                    )
                    if self._log_clipping_warnings:
                        msg = (
                            f"Left arm joint {idx+1} target {left_arm_target_q[idx]} out of range "
                            f"[{self.left_arm_joint_low[idx]}, {self.left_arm_joint_high[idx]}]. Clipped."
                        )
                        self.get_logger().warning(msg)
                else:
                    msg = (
                        f"Left arm joint {idx+1} target {left_arm_target_q[idx]} out of range "
                        f"[{self.left_arm_joint_low[idx]}, {self.left_arm_joint_high[idx]}]."
                    )
                    raise ValueError(msg)

        if right_arm_target_q is not None:
            right_in_range = np.logical_and(
                right_arm_target_q >= self.right_arm_joint_low,
                right_arm_target_q <= self.right_arm_joint_high,
            )
            for idx in np.where(~right_in_range)[0]:
                if self._on_arm_cmd_out_of_range == "clip":
                    right_arm_target_q[idx] = np.clip(
                        right_arm_target_q[idx],
                        self.right_arm_joint_low[idx],
                        self.right_arm_joint_high[idx],
                    )
                    if self._log_clipping_warnings:
                        msg = (
                            f"Right arm joint {idx+1} target {right_arm_target_q[idx]} out of range "
                            f"[{self.right_arm_joint_low[idx]}, {self.right_arm_joint_high[idx]}]. Clipped."
                        )
                        self.get_logger().warning(msg)
                else:
                    msg = (
                        f"Right arm joint {idx+1} target {right_arm_target_q[idx]} out of range "
                        f"[{self.right_arm_joint_low[idx]}, {self.right_arm_joint_high[idx]}]."
                    )
                    raise ValueError(msg)

        if torso_target_q is not None:
            torso_in_range = np.logical_and(
                torso_target_q >= self.torso_joint_low,
                torso_target_q <= self.torso_joint_high,
            )
            for idx in np.where(~torso_in_range)[0]:
                if self._on_torso_cmd_out_of_range == "clip":
                    torso_target_q[idx] = np.clip(
                        torso_target_q[idx],
                        self.torso_joint_low[idx],
                        self.torso_joint_high[idx],
                    )
                    if self._log_clipping_warnings:
                        msg = (
                            f"Torso joint {idx+1} target {torso_target_q[idx]} out of range "
                            f"[{self.torso_joint_low[idx]}, {self.torso_joint_high[idx]}]. Clipped."
                        )
                        self.get_logger().warning(msg)
                else:
                    msg = (
                        f"Torso joint {idx+1} target {torso_target_q[idx]} out of range "
                        f"[{self.torso_joint_low[idx]}, {self.torso_joint_high[idx]}]."
                    )
                    raise ValueError(msg)

        # compute step counts
        last_two_arms_joint_positions = self.last_joint_position
        left_arm_joint_state = JointState()
        left_arm_joint_state.position = last_two_arms_joint_positions[
            "left_arm"
        ].tolist()
        left_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                left_arm_target_q
                                - np.array(left_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if left_arm_target_q is not None
            else 0
        )
        right_arm_joint_state = JointState()
        right_arm_joint_state.position = last_two_arms_joint_positions[
            "right_arm"
        ].tolist()
        right_arm_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (
                                right_arm_target_q
                                - np.array(right_arm_joint_state.position)
                            )
                            / self._arm_joint_control_step_interval
                        )
                    )
                )
            )
            if right_arm_target_q is not None
            else 0
        )
        last_torso_joint_positions = self.last_joint_position["torso"]
        torso_joint_state = JointState()
        torso_joint_state.position = last_torso_joint_positions.tolist()
        torso_n_steps = (
            int(
                np.ceil(
                    np.max(
                        np.abs(
                            (torso_target_q - np.array(torso_joint_state.position))
                            / self._torso_joint_control_step_interval
                        )
                    )
                )
            )
            if torso_target_q is not None
            else 0
        )

        # TORSO: Publish immediately without interpolation for responsive control
        # This ensures holding the torso button results in continuous movement
        if torso_target_q is not None:
            torso_joint_state.header.stamp = self._now_msg()
            torso_joint_state.position = torso_target_q.tolist()
            self._torso_joint_target_position_pub.publish(torso_joint_state)

        # ARMS: Use interpolation loop only for arms (torso already published above)
        max_arm_n_steps = max(left_arm_n_steps, right_arm_n_steps)
        if max_arm_n_steps > 1:
            left_arm_increment = (
                (
                    (left_arm_target_q - np.array(left_arm_joint_state.position))
                    / (left_arm_n_steps - 1)
                )
                if (left_arm_target_q is not None and left_arm_n_steps > 1)
                else (
                    (left_arm_target_q - np.array(left_arm_joint_state.position))
                    if left_arm_target_q is not None
                    else np.zeros(7)
                )
            )
            right_arm_increment = (
                (
                    (right_arm_target_q - np.array(right_arm_joint_state.position))
                    / (right_arm_n_steps - 1)
                )
                if (right_arm_target_q is not None and right_arm_n_steps > 1)
                else (
                    (right_arm_target_q - np.array(right_arm_joint_state.position))
                    if right_arm_target_q is not None
                    else np.zeros(7)
                )
            )

            for step in range(max_arm_n_steps - 1):
                if left_arm_target_q is not None:
                    left_arm_joint_state.header.stamp = self._now_msg()
                    if step <= left_arm_n_steps - 1:
                        left_arm_joint_state.position = (
                            np.array(left_arm_joint_state.position) + left_arm_increment
                        ).tolist()
                    else:
                        left_arm_joint_state.position = left_arm_target_q.tolist()
                    self._left_arm_joint_target_position_pub.publish(
                        left_arm_joint_state
                    )

                if right_arm_target_q is not None:
                    right_arm_joint_state.header.stamp = self._now_msg()
                    if step <= right_arm_n_steps - 1:
                        right_arm_joint_state.position = (
                            np.array(right_arm_joint_state.position)
                            + right_arm_increment
                        ).tolist()
                    else:
                        right_arm_joint_state.position = right_arm_target_q.tolist()
                    self._right_arm_joint_target_position_pub.publish(
                        right_arm_joint_state
                    )

                self._sleep_tick()

        # ensure exact final target positions for arms
        if left_arm_target_q is not None:
            left_arm_joint_state.header.stamp = self._now_msg()
            left_arm_joint_state.position = left_arm_target_q.tolist()
            self._left_arm_joint_target_position_pub.publish(left_arm_joint_state)
        if right_arm_target_q is not None:
            right_arm_joint_state.header.stamp = self._now_msg()
            right_arm_joint_state.position = right_arm_target_q.tolist()
            self._right_arm_joint_target_position_pub.publish(right_arm_joint_state)

        if not (
            left_arm_target_q is None
            and right_arm_target_q is None
            and torso_target_q is None
        ):
            self._sleep_tick()

    def _gripper_control(self, gripper_action: Dict[str, float]):
        left_gripper_action = gripper_action["left"]
        right_gripper_action = gripper_action["right"]
        assert (
            1 >= left_gripper_action >= 0
        ), "Invalid left gripper action, must be between 0 and 1"
        assert (
            1 >= right_gripper_action >= 0
        ), "Invalid right gripper action, must be between 0 and 1"
        if self._left_gripper is not None:
            self._left_gripper.act(left_gripper_action)
        if self._right_gripper is not None:
            self._right_gripper.act(right_gripper_action)

    def _rgb_callback(self, rgb_msg: CompressedImage, name: str):
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        img = np.asarray(
            self._cv_bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        )
        self._rgb[name] = {"img": img, "stamp": timestamp}

    def _left_arm_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:7]]),
                self.left_arm_joint_low,
                self.left_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:7]),
            "joint_effort": np.array([data.effort][:7]),
            "seq": np.array(
                [data.header.stamp.nanosec]
            ),  # seq not present in ROS 2; store something monotonic-ish
            "stamp": np.array(
                [data.header.stamp.sec + data.header.stamp.nanosec * 1e-9]
            ),
        }
        if self._left_arm_joint_state_buffer is None:
            self._left_arm_joint_state_buffer = new_state
        else:
            self._left_arm_joint_state_buffer = U.any_concat(
                [self._left_arm_joint_state_buffer, new_state], dim=0
            )
            self._left_arm_joint_state_buffer = U.any_slice(
                self._left_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _right_arm_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:7]]),
                self.right_arm_joint_low,
                self.right_arm_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:7]),
            "joint_effort": np.array([data.effort][:7]),
            "seq": np.array([data.header.stamp.nanosec]),
            "stamp": np.array(
                [data.header.stamp.sec + data.header.stamp.nanosec * 1e-9]
            ),
        }
        if self._right_arm_joint_state_buffer is None:
            self._right_arm_joint_state_buffer = new_state
        else:
            self._right_arm_joint_state_buffer = U.any_concat(
                [self._right_arm_joint_state_buffer, new_state], dim=0
            )
            self._right_arm_joint_state_buffer = U.any_slice(
                self._right_arm_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _torso_state_callback(self, data: JointState):
        new_state = {
            "joint_position": np.clip(
                np.array([data.position[:4]]),
                self.torso_joint_low,
                self.torso_joint_high,
            ),
            "joint_velocity": np.array([data.velocity][:4]),
            "joint_effort": np.array([data.effort][:4]),
            "seq": np.array([data.header.stamp.nanosec]),
            "stamp": np.array(
                [data.header.stamp.sec + data.header.stamp.nanosec * 1e-9]
            ),
        }
        if self._torso_joint_state_buffer is None:
            self._torso_joint_state_buffer = new_state
        else:
            self._torso_joint_state_buffer = U.any_concat(
                [self._torso_joint_state_buffer, new_state], dim=0
            )
            self._torso_joint_state_buffer = U.any_slice(
                self._torso_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    def _chassis_state_callback(self, data: JointState):
        # Chassis feedback: position has 3 values, velocity has 6 values (use first 3)
        new_state = {
            "joint_position": np.array([data.position[:3]]),
            "joint_velocity": np.array([data.velocity[:3]]),
            "seq": np.array([data.header.stamp.nanosec]),
            "stamp": np.array(
                [data.header.stamp.sec + data.header.stamp.nanosec * 1e-9]
            ),
        }
        if self._chassis_joint_state_buffer is None:
            self._chassis_joint_state_buffer = new_state
        else:
            self._chassis_joint_state_buffer = U.any_concat(
                [self._chassis_joint_state_buffer, new_state], dim=0
            )
            self._chassis_joint_state_buffer = U.any_slice(
                self._chassis_joint_state_buffer, np.s_[-self._state_buffer_size :]
            )

    # ---------- Lifecycle ----------
    def close(self):
        # stop the base
        self.stop_mobile_base()
        if self._left_gripper is not None:
            self._left_gripper.close()
        if self._right_gripper is not None:
            self._right_gripper.close()
        try:
            self.destroy_node()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def last_joint_position(self) -> Dict[str, np.ndarray]:
        if (
            self._left_arm_joint_state_buffer is None
            or self._right_arm_joint_state_buffer is None
            or self._torso_joint_state_buffer is None
        ):
            return None  # return None if any buffer is None
        return {
            "left_arm": U.any_slice(self._left_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "right_arm": U.any_slice(self._right_arm_joint_state_buffer, -1)[
                "joint_position"
            ],
            "torso": U.any_slice(self._torso_joint_state_buffer, -1)["joint_position"],
        }

    @property
    def last_gripper_state(self):
        if (
            self._left_gripper.state_buffer is None
            or self._right_gripper.state_buffer is None
        ):
            return None  # return None if any buffer is None
        return {
            "left_gripper": (U.any_slice(self._left_gripper.state_buffer, -1)),
            "right_gripper": (U.any_slice(self._right_gripper.state_buffer, -1)),
        }

    @property
    def last_chassis_position(self) -> Optional[np.ndarray]:
        """Get the last chassis position (3 values) from /hdas/feedback_chassis."""
        if self._chassis_joint_state_buffer is None:
            return None
        return U.any_slice(self._chassis_joint_state_buffer, -1)["joint_position"]

    @property
    def last_chassis_velocity(self) -> Optional[np.ndarray]:
        """Get the last chassis velocity (first 3 of 6 values) from /hdas/feedback_chassis."""
        if self._chassis_joint_state_buffer is None:
            return None
        return U.any_slice(self._chassis_joint_state_buffer, -1)["joint_velocity"]

    @property
    def last_joint_velocity(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the last joint velocities for all robot parts."""
        if (
            self._left_arm_joint_state_buffer is None
            or self._right_arm_joint_state_buffer is None
            or self._torso_joint_state_buffer is None
        ):
            return None
        return {
            "left_arm": U.any_slice(self._left_arm_joint_state_buffer, -1)["joint_velocity"],
            "right_arm": U.any_slice(self._right_arm_joint_state_buffer, -1)["joint_velocity"],
            "torso": U.any_slice(self._torso_joint_state_buffer, -1)["joint_velocity"],
        }

    @property
    def last_rgb(self):
        # Because v can be None
        if self._rgb is not None:
            return_dict = {}
            for k, v in self._rgb.items():
                if v is not None:
                    return_dict[k] = {
                        "img": v["img"],
                        "stamp": v["stamp"],
                    }
                else:
                    return_dict = None  # set to None if any key is not ready
                    break
            return return_dict
        else:
            return None

    @property
    def curr_base_pose(self):
        return self._odom.curr_base_pose

    @property
    def curr_base_position(self):
        return self._odom.curr_base_position

    @property
    def curr_base_orientation(self):
        return self._odom.curr_base_orientation

    @property
    def curr_base_velocity(self):
        return self._odom.curr_base_velocity
