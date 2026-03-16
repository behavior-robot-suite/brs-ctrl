from typing import Optional
import os
import threading

import yaml
import numpy as np
from pyjoycon import JoyCon, get_R_id, get_L_id

try:
    import rospy
except ImportError as e:
    print(f"Failed to import ROS related modules, R1JoyConInterface won't work.")
    print(e)
from std_msgs.msg import Bool


class R1JoyConInterface:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])

    def __init__(
        self,
        *,
        calibration_file: Optional[str] = None,
        # ====== mobile base ======
        mobile_base_x_move_max: float = 0.3,
        mobile_base_y_move_max: float = 0.3,
        mobile_base_yaw_rotate_max: float = 0.4,
        # ====== torso ======
        torso_joint_max_delta: float = 0.1,
        torso_joints1_2_stand_q: np.ndarray = np.array([0, 0, 0]),
        torso_joints1_2_squat_q: np.ndarray = np.array([1.74, -2.70, -0.96]),
        # ====== gripper ======
        gripper_toggle_mode: bool = False,
        # ====== common ======
        joystick_ema_alpha: float = 0.9,
        joystick_idle_percentage: float = 0.2,
        # ====== publish functional buttons ======
        ros_publish_functional_buttons: bool = False,
        init_ros_node: bool = True,
        publish_topic: str = "/joycon/functional_buttons",
        publish_freq: float = 10,
    ):
        if calibration_file is None:
            calibration_file = os.path.join(
                os.path.dirname(__file__), "joycon_calibration.yaml"
            )
        assert os.path.exists(
            calibration_file
        ), f"Calibration file not found: {calibration_file}"
        with open(calibration_file, "r") as f:
            self.calibration_data = yaml.load(f, Loader=yaml.FullLoader)["joystick"]
        assert (
            0 <= joystick_idle_percentage <= 1
        ), "Joystick idle percentage must be in [0, 1]."
        full_lh_range = (
            self.calibration_data["left"]["horizontal"]["limits"][1]
            - self.calibration_data["left"]["horizontal"]["limits"][0]
        )
        full_lv_range = (
            self.calibration_data["left"]["vertical"]["limits"][1]
            - self.calibration_data["left"]["vertical"]["limits"][0]
        )
        full_rh_range = (
            self.calibration_data["right"]["horizontal"]["limits"][1]
            - self.calibration_data["right"]["horizontal"]["limits"][0]
        )
        full_rv_range = (
            self.calibration_data["right"]["vertical"]["limits"][1]
            - self.calibration_data["right"]["vertical"]["limits"][0]
        )
        self.calibration_data["left"]["horizontal"]["center_range"][0] -= (
            full_lh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["left"]["horizontal"]["center_range"][1] += (
            full_lh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["left"]["vertical"]["center_range"][0] -= (
            full_lv_range * joystick_idle_percentage / 2
        )
        self.calibration_data["left"]["vertical"]["center_range"][1] += (
            full_lv_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["horizontal"]["center_range"][0] -= (
            full_rh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["horizontal"]["center_range"][1] += (
            full_rh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["vertical"]["center_range"][0] -= (
            full_rv_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["vertical"]["center_range"][1] += (
            full_rv_range * joystick_idle_percentage / 2
        )

        self._lh_unpressed_ema = None
        self._lh_pressed_ema = None
        self._lv_ema = None
        self._rh_ema = None
        self._rv_ema = None
        self._mobile_base_x_move_max = mobile_base_x_move_max
        self._mobile_base_y_move_max = mobile_base_y_move_max
        self._mobile_base_yaw_rotate_max = mobile_base_yaw_rotate_max
        self._torso_joint_max_delta = torso_joint_max_delta
        self._torso_joints1_2_stand_q = torso_joints1_2_stand_q
        self._torso_joints1_2_squat_q = torso_joints1_2_squat_q
        self._gripper_toggle_mode = gripper_toggle_mode
        self._left_gripper_current_action = self._right_gripper_current_action = 0.1
        self._left_gripper_button_pressed_times = (
            self._right_gripper_button_pressed_times
        ) = 0
        self._joystick_ema_alpha = joystick_ema_alpha

        jc_id_left = get_L_id()
        jc_id_right = get_R_id()
        assert jc_id_left[0] is not None, "Failed to connect to Left JoyCon!"
        assert jc_id_right[0] is not None, "Failed to connect to Right JoyCon!"
        self.jc_left = JoyCon(*jc_id_left)
        self.jc_right = JoyCon(*jc_id_right)

        jc_l_battery = self.jc_left.get_battery_level()
        jc_r_battery = self.jc_right.get_battery_level()
        if jc_l_battery <= 1:
            print(f"[WARN] Left JoyCon battery is low: {jc_l_battery}.")
        if jc_r_battery <= 1:
            print(f"[WARN] Right JoyCon battery is low: {jc_r_battery}.")

        self._ros_publish_functional_buttons = ros_publish_functional_buttons
        if ros_publish_functional_buttons:
            # init ros node
            if init_ros_node:
                rospy.init_node("joycon")
            self._rate = rospy.Rate(publish_freq)
            self._jc_r_x_button_pub = rospy.Publisher(
                publish_topic + "/x_button", Bool, queue_size=10
            )
            self._jc_r_y_button_pub = rospy.Publisher(
                publish_topic + "/y_button", Bool, queue_size=10
            )
            self._jc_r_a_button_pub = rospy.Publisher(
                publish_topic + "/a_button", Bool, queue_size=10
            )
            self._jc_r_b_button_pub = rospy.Publisher(
                publish_topic + "/b_button", Bool, queue_size=10
            )

            self._stop_pub_event = threading.Event()
            # start thread
            self._stop_pub_event.clear()
            self._pub_thread = threading.Thread(
                target=self._functional_buttons_ros_pub_loop, daemon=True
            )
            self._pub_thread.start()
            rospy.loginfo(
                f"Publishing JoyCon functional buttons to ROS topic {publish_topic}."
            )

    def act(self, curr_torso_q: np.ndarray):
        """
        Mapping strategy:
            Mobile base: When left joystick is not pressed, map left joystick to xy translation; when left joystick is
                pressed, map left joystick horizontal value to base yaw rotation.
            Torso: Map right joystick vertical value to torso forward/backward movement; Map right joystick horizontal
                value to torso yaw rotation. Map left joycon up button to torso standing and down button to
                torso squatting.
            Gripper: Map ZL/ZR to left/right gripper.
        """
        assert curr_torso_q.shape == self.torso_joint_high.shape

        # read joystick values
        is_left_joystick_pressed = self.jc_left.get_button_l_stick()
        raw_lh = self.jc_left.get_stick_left_horizontal()
        raw_lv = self.jc_left.get_stick_left_vertical()
        lh_neutral = 0.5 * sum(
            self.calibration_data["left"]["horizontal"]["center_range"]
        )
        lv_neutral = 0.5 * sum(
            self.calibration_data["left"]["vertical"]["center_range"]
        )
        if is_left_joystick_pressed:
            self._lh_pressed_ema = (
                raw_lh
                if self._lh_pressed_ema is None
                else self._joystick_ema_alpha * raw_lh
                + (1 - self._joystick_ema_alpha) * self._lh_pressed_ema
            )
            self._lh_unpressed_ema = (
                lh_neutral
                if self._lh_unpressed_ema is None
                else self._joystick_ema_alpha * lh_neutral
                + (1 - self._joystick_ema_alpha) * self._lh_unpressed_ema
            )
            self._lv_ema = (
                lv_neutral
                if self._lv_ema is None
                else self._joystick_ema_alpha * lv_neutral
                + (1 - self._joystick_ema_alpha) * self._lv_ema
            )
        else:
            self._lh_pressed_ema = (
                lh_neutral
                if self._lh_pressed_ema is None
                else self._joystick_ema_alpha * lh_neutral
                + (1 - self._joystick_ema_alpha) * self._lh_pressed_ema
            )
            self._lh_unpressed_ema = (
                raw_lh
                if self._lh_unpressed_ema is None
                else self._joystick_ema_alpha * raw_lh
                + (1 - self._joystick_ema_alpha) * self._lh_unpressed_ema
            )
            self._lv_ema = (
                raw_lv
                if self._lv_ema is None
                else self._joystick_ema_alpha * raw_lv
                + (1 - self._joystick_ema_alpha) * self._lv_ema
            )

        raw_rh = self.jc_right.get_stick_right_horizontal()
        self._rh_ema = (
            raw_rh
            if self._rh_ema is None
            else self._joystick_ema_alpha * raw_rh
            + (1 - self._joystick_ema_alpha) * self._rh_ema
        )
        raw_rv = self.jc_right.get_stick_right_vertical()
        self._rv_ema = (
            raw_rv
            if self._rv_ema is None
            else self._joystick_ema_alpha * raw_rv
            + (1 - self._joystick_ema_alpha) * self._rv_ema
        )

        # process mobile base movement
        lh_neutral_range = self.calibration_data["left"]["horizontal"]["center_range"]
        lh_limits = self.calibration_data["left"]["horizontal"]["limits"]
        lv_neutral_range = self.calibration_data["left"]["vertical"]["center_range"]
        lv_limits = self.calibration_data["left"]["vertical"]["limits"]
        # if pressed, map to yaw rotation only
        if is_left_joystick_pressed:
            lh = self._lh_pressed_ema
            if lh_neutral_range[0] <= lh <= lh_neutral_range[1]:
                base_displacement = [0, 0, 0]
            else:
                lh = np.clip(lh, lh_limits[0], lh_limits[1])
                delta_lh = lh - 0.5 * sum(lh_neutral_range)
                delta_lh *= -1  # because horizontal direction is reversed
                delta_lh = np.sign(delta_lh) * np.clip(
                    abs(delta_lh), a_max=(lh_limits[1] - lh_limits[0]) / 2, a_min=0
                )
                base_displacement = [
                    0,
                    0,
                    2.0
                    * delta_lh
                    / (lh_limits[1] - lh_limits[0])
                    * self._mobile_base_yaw_rotate_max,
                ]
        else:
            # map to xy translation
            lh = self._lh_unpressed_ema
            lv = self._lv_ema
            base_displacement = [0, 0, 0]
            if lh > lh_neutral_range[1] or lh < lh_neutral_range[0]:
                lh = np.clip(lh, lh_limits[0], lh_limits[1])
                delta_lh = lh - 0.5 * sum(lh_neutral_range)
                delta_lh *= -1  # because horizontal direction is reversed
                delta_lh = np.sign(delta_lh) * np.clip(
                    abs(delta_lh), a_max=(lh_limits[1] - lh_limits[0]) / 2, a_min=0
                )
                base_displacement[1] = (
                    2.0
                    * delta_lh
                    / (lh_limits[1] - lh_limits[0])
                    * self._mobile_base_x_move_max
                )
            if lv > lv_neutral_range[1] or lv < lv_neutral_range[0]:
                lv = np.clip(lv, lv_limits[0], lv_limits[1])
                delta_lv = lv - 0.5 * sum(lv_neutral_range)
                delta_lv = np.sign(delta_lv) * np.clip(
                    abs(delta_lv), a_max=(lv_limits[1] - lv_limits[0]) / 2, a_min=0
                )
                base_displacement[0] = (
                    2.0
                    * delta_lv
                    / (lv_limits[1] - lv_limits[0])
                    * self._mobile_base_y_move_max
                )

        # process torso movement
        rh_neutral_range = self.calibration_data["right"]["horizontal"]["center_range"]
        rh_limits = self.calibration_data["right"]["horizontal"]["limits"]
        rv_neutral_range = self.calibration_data["right"]["vertical"]["center_range"]
        rv_limits = self.calibration_data["right"]["vertical"]["limits"]
        rh = self._rh_ema
        rv = self._rv_ema

        torso_cmd = curr_torso_q.copy()  # Must copy to avoid modifying input array
        if rh > rh_neutral_range[1] or rh < rh_neutral_range[0]:
            rh = np.clip(rh, rh_limits[0], rh_limits[1])
            delta_rh = rh - 0.5 * sum(rh_neutral_range)
            delta_rh *= -1  # because horizontal direction is reversed
            delta_rh = np.sign(delta_rh) * np.clip(
                abs(delta_rh), a_max=(rh_limits[1] - rh_limits[0]) / 2, a_min=0
            )
            torso_yaw_rotation = (
                2.0
                * delta_rh
                / (rh_limits[1] - rh_limits[0])
                * self._torso_joint_max_delta
            )
            torso_cmd[3] += torso_yaw_rotation
        if rv > rv_neutral_range[1] or rv < rv_neutral_range[0]:
            rv = np.clip(rv, rv_limits[0], rv_limits[1])
            delta_rv = rv - 0.5 * sum(rv_neutral_range)
            delta_rv = np.sign(delta_rv) * np.clip(
                abs(delta_rv), a_max=(rv_limits[1] - rv_limits[0]) / 2, a_min=0
            )
            delta_rv *= -1  # because vertical direction is reversed
            torso_cmd[2] += (
                2
                * delta_rv
                / (rv_limits[1] - rv_limits[0])
                * self._torso_joint_max_delta
            )
        if self.jc_left.get_button_up():
            diff = self._torso_joints1_2_stand_q - torso_cmd[:3]
            torso_cmd[0] += np.sign(diff[0]) * self._torso_joint_max_delta * 1.74 / 2.7
            torso_cmd[1] += np.sign(diff[1]) * self._torso_joint_max_delta
            torso_cmd[2] += np.sign(diff[2]) * self._torso_joint_max_delta * 0.96 / 2.7
        elif self.jc_left.get_button_down():
            diff = self._torso_joints1_2_squat_q - torso_cmd[:3]
            torso_cmd[0] += np.sign(diff[0]) * self._torso_joint_max_delta * 1.74 / 2.7
            torso_cmd[1] += np.sign(diff[1]) * self._torso_joint_max_delta
            torso_cmd[2] += np.sign(diff[2]) * self._torso_joint_max_delta * 0.96 / 2.7
        torso_cmd[0] = np.clip(
            torso_cmd[0],
            self._torso_joints1_2_stand_q[0],
            self._torso_joints1_2_squat_q[0],
        )
        torso_cmd[1] = np.clip(
            torso_cmd[1],
            self._torso_joints1_2_squat_q[1],
            self._torso_joints1_2_stand_q[1],
        )
        torso_cmd[2] = np.clip(
            torso_cmd[2],
            self._torso_joints1_2_squat_q[2],
            self._torso_joints1_2_stand_q[2],
        )
        torso_cmd = np.clip(torso_cmd, self.torso_joint_low, self.torso_joint_high)

        # process gripper
        if self._gripper_toggle_mode:
            if self.jc_left.get_button_zl():
                if self._left_gripper_button_pressed_times >= 30:
                    left_gripper = (
                        1.0 if self._left_gripper_current_action < 0.5 else 0.1
                    )
                    self._left_gripper_current_action = left_gripper
                    self._left_gripper_button_pressed_times = 0
                else:
                    left_gripper = self._left_gripper_current_action
                    self._left_gripper_button_pressed_times += 1
            else:
                left_gripper = self._left_gripper_current_action

            if self.jc_right.get_button_zr():
                if self._right_gripper_button_pressed_times >= 30:
                    right_gripper = (
                        1.0 if self._right_gripper_current_action < 0.5 else 0.1
                    )
                    self._right_gripper_current_action = right_gripper
                    self._right_gripper_button_pressed_times = 0
                else:
                    right_gripper = self._right_gripper_current_action
                    self._right_gripper_button_pressed_times += 1
            else:
                right_gripper = self._right_gripper_current_action
        else:
            left_gripper = 1.0 if self.jc_left.get_button_zl() else 0.1
            right_gripper = 1.0 if self.jc_right.get_button_zr() else 0.1

        return {
            "mobile_base_cmd": np.array(base_displacement),
            "torso_controller": "joint_position",
            "torso_cmd": torso_cmd,
            "gripper_cmd": {
                "left": left_gripper,
                "right": right_gripper,
            },
        }

    def _functional_buttons_ros_pub_loop(self):
        while not rospy.is_shutdown() and not self._stop_pub_event.is_set():
            self._jc_r_x_button_pub.publish(Bool(self.jc_right.get_button_x()))
            self._jc_r_y_button_pub.publish(Bool(self.jc_right.get_button_y()))
            self._jc_r_a_button_pub.publish(Bool(self.jc_right.get_button_a()))
            self._jc_r_b_button_pub.publish(Bool(self.jc_right.get_button_b()))
            self._rate.sleep()

    def __del__(self):
        del self.jc_left
        del self.jc_right
        if self._ros_publish_functional_buttons:
            try:
                self._stop_pub_event.set()
                self._pub_thread.join()
                rospy.loginfo("JoyCon functional buttons publisher stopped.")
                rospy.signal_shutdown("JoyConInterface is deleted.")
            except:
                pass

    def close(self):
        self.__del__()


class R1ProJoyConInterface:
    torso_joint_high = np.array([1.8326, 2.5307, 1.5708, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -1.8326, -3.0543])

    def __init__(
        self,
        *,
        calibration_file: Optional[str] = None,
        # ====== mobile base ======
        mobile_base_x_move_max: float = 0.4,
        mobile_base_y_move_max: float = 0.4,
        mobile_base_yaw_rotate_max: float = 0.4,
        # ====== torso ======
        torso_joint_max_delta: float = 0.1,
        torso_joints1_2_stand_q: np.ndarray = np.array([0, 0, 0]),
        torso_joints1_2_squat_q: np.ndarray = np.array([1.74, -2.70, -0.96]),
        # ====== gripper ======
        gripper_toggle_mode: bool = False,
        # ====== common ======
        joystick_ema_alpha: float = 0.9,
        joystick_idle_percentage: float = 0.2,
    ):
        if calibration_file is None:
            calibration_file = os.path.join(
                os.path.dirname(__file__), "joycon_calibration.yaml"
            )
        assert os.path.exists(
            calibration_file
        ), f"Calibration file not found: {calibration_file}"
        with open(calibration_file, "r") as f:
            self.calibration_data = yaml.load(f, Loader=yaml.FullLoader)["joystick"]
        assert (
            0 <= joystick_idle_percentage <= 1
        ), "Joystick idle percentage must be in [0, 1]."
        full_lh_range = (
            self.calibration_data["left"]["horizontal"]["limits"][1]
            - self.calibration_data["left"]["horizontal"]["limits"][0]
        )
        full_lv_range = (
            self.calibration_data["left"]["vertical"]["limits"][1]
            - self.calibration_data["left"]["vertical"]["limits"][0]
        )
        full_rh_range = (
            self.calibration_data["right"]["horizontal"]["limits"][1]
            - self.calibration_data["right"]["horizontal"]["limits"][0]
        )
        full_rv_range = (
            self.calibration_data["right"]["vertical"]["limits"][1]
            - self.calibration_data["right"]["vertical"]["limits"][0]
        )
        self.calibration_data["left"]["horizontal"]["center_range"][0] -= (
            full_lh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["left"]["horizontal"]["center_range"][1] += (
            full_lh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["left"]["vertical"]["center_range"][0] -= (
            full_lv_range * joystick_idle_percentage / 2
        )
        self.calibration_data["left"]["vertical"]["center_range"][1] += (
            full_lv_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["horizontal"]["center_range"][0] -= (
            full_rh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["horizontal"]["center_range"][1] += (
            full_rh_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["vertical"]["center_range"][0] -= (
            full_rv_range * joystick_idle_percentage / 2
        )
        self.calibration_data["right"]["vertical"]["center_range"][1] += (
            full_rv_range * joystick_idle_percentage / 2
        )

        self._lh_unpressed_ema = None
        self._lh_pressed_ema = None
        self._lv_ema = None
        self._rh_ema = None
        self._rv_ema = None
        self._mobile_base_x_move_max = mobile_base_x_move_max
        self._mobile_base_y_move_max = mobile_base_y_move_max
        self._mobile_base_yaw_rotate_max = mobile_base_yaw_rotate_max
        self._torso_joint_max_delta = torso_joint_max_delta
        self._torso_joints1_2_stand_q = torso_joints1_2_stand_q
        self._torso_joints1_2_squat_q = torso_joints1_2_squat_q
        self._gripper_toggle_mode = gripper_toggle_mode
        self._left_gripper_current_action = self._right_gripper_current_action = 0.1
        self._left_gripper_button_pressed_times = (
            self._right_gripper_button_pressed_times
        ) = 0
        self._joystick_ema_alpha = joystick_ema_alpha

        jc_id_left = get_L_id()
        jc_id_right = get_R_id()
        assert jc_id_left[0] is not None, "Failed to connect to Left JoyCon!"
        assert jc_id_right[0] is not None, "Failed to connect to Right JoyCon!"
        self.jc_left = JoyCon(*jc_id_left)
        self.jc_right = JoyCon(*jc_id_right)

        jc_l_battery = self.jc_left.get_battery_level()
        jc_r_battery = self.jc_right.get_battery_level()
        if jc_l_battery <= 1:
            print(f"[WARN] Left JoyCon battery is low: {jc_l_battery}.")
        if jc_r_battery <= 1:
            print(f"[WARN] Right JoyCon battery is low: {jc_r_battery}.")

    def act(self, curr_torso_q: np.ndarray):
        """
        Mapping strategy:
            Mobile base: When left joystick is not pressed, map left joystick to xy translation; when left joystick is
                pressed, map left joystick horizontal value to base yaw rotation.
            Torso: Map right joystick vertical value to torso forward/backward movement; Map right joystick horizontal
                value to torso yaw rotation. Map left joycon up button to torso standing and down button to
                torso squatting.
            Gripper: Map ZL/ZR to left/right gripper.
        """
        assert curr_torso_q.shape == self.torso_joint_high.shape

        # read joystick values
        is_left_joystick_pressed = self.jc_left.get_button_l_stick()
        raw_lh = self.jc_left.get_stick_left_horizontal()
        raw_lv = self.jc_left.get_stick_left_vertical()
        lh_neutral = 0.5 * sum(
            self.calibration_data["left"]["horizontal"]["center_range"]
        )
        lv_neutral = 0.5 * sum(
            self.calibration_data["left"]["vertical"]["center_range"]
        )
        if is_left_joystick_pressed:
            self._lh_pressed_ema = (
                raw_lh
                if self._lh_pressed_ema is None
                else self._joystick_ema_alpha * raw_lh
                + (1 - self._joystick_ema_alpha) * self._lh_pressed_ema
            )
            self._lh_unpressed_ema = (
                lh_neutral
                if self._lh_unpressed_ema is None
                else self._joystick_ema_alpha * lh_neutral
                + (1 - self._joystick_ema_alpha) * self._lh_unpressed_ema
            )
            self._lv_ema = (
                lv_neutral
                if self._lv_ema is None
                else self._joystick_ema_alpha * lv_neutral
                + (1 - self._joystick_ema_alpha) * self._lv_ema
            )
        else:
            self._lh_pressed_ema = (
                lh_neutral
                if self._lh_pressed_ema is None
                else self._joystick_ema_alpha * lh_neutral
                + (1 - self._joystick_ema_alpha) * self._lh_pressed_ema
            )
            self._lh_unpressed_ema = (
                raw_lh
                if self._lh_unpressed_ema is None
                else self._joystick_ema_alpha * raw_lh
                + (1 - self._joystick_ema_alpha) * self._lh_unpressed_ema
            )
            self._lv_ema = (
                raw_lv
                if self._lv_ema is None
                else self._joystick_ema_alpha * raw_lv
                + (1 - self._joystick_ema_alpha) * self._lv_ema
            )

        raw_rh = self.jc_right.get_stick_right_horizontal()
        self._rh_ema = (
            raw_rh
            if self._rh_ema is None
            else self._joystick_ema_alpha * raw_rh
            + (1 - self._joystick_ema_alpha) * self._rh_ema
        )
        raw_rv = self.jc_right.get_stick_right_vertical()
        self._rv_ema = (
            raw_rv
            if self._rv_ema is None
            else self._joystick_ema_alpha * raw_rv
            + (1 - self._joystick_ema_alpha) * self._rv_ema
        )

        # process mobile base movement
        lh_neutral_range = self.calibration_data["left"]["horizontal"]["center_range"]
        lh_limits = self.calibration_data["left"]["horizontal"]["limits"]
        lv_neutral_range = self.calibration_data["left"]["vertical"]["center_range"]
        lv_limits = self.calibration_data["left"]["vertical"]["limits"]
        # if pressed, map to yaw rotation only
        if is_left_joystick_pressed:
            lh = self._lh_pressed_ema
            if lh_neutral_range[0] <= lh <= lh_neutral_range[1]:
                base_displacement = [0, 0, 0]
            else:
                lh = np.clip(lh, lh_limits[0], lh_limits[1])
                delta_lh = lh - 0.5 * sum(lh_neutral_range)
                delta_lh *= -1  # because horizontal direction is reversed
                delta_lh = np.sign(delta_lh) * np.clip(
                    abs(delta_lh), a_max=(lh_limits[1] - lh_limits[0]) / 2, a_min=0
                )
                base_displacement = [
                    0,
                    0,
                    2.0
                    * delta_lh
                    / (lh_limits[1] - lh_limits[0])
                    * self._mobile_base_yaw_rotate_max,
                ]
        else:
            # map to xy translation
            lh = self._lh_unpressed_ema
            lv = self._lv_ema
            base_displacement = [0, 0, 0]
            if lh > lh_neutral_range[1] or lh < lh_neutral_range[0]:
                lh = np.clip(lh, lh_limits[0], lh_limits[1])
                delta_lh = lh - 0.5 * sum(lh_neutral_range)
                delta_lh *= -1  # because horizontal direction is reversed
                delta_lh = np.sign(delta_lh) * np.clip(
                    abs(delta_lh), a_max=(lh_limits[1] - lh_limits[0]) / 2, a_min=0
                )
                base_displacement[1] = (
                    2.0
                    * delta_lh
                    / (lh_limits[1] - lh_limits[0])
                    * self._mobile_base_x_move_max
                )
            if lv > lv_neutral_range[1] or lv < lv_neutral_range[0]:
                lv = np.clip(lv, lv_limits[0], lv_limits[1])
                delta_lv = lv - 0.5 * sum(lv_neutral_range)
                delta_lv = np.sign(delta_lv) * np.clip(
                    abs(delta_lv), a_max=(lv_limits[1] - lv_limits[0]) / 2, a_min=0
                )
                base_displacement[0] = (
                    2.0
                    * delta_lv
                    / (lv_limits[1] - lv_limits[0])
                    * self._mobile_base_y_move_max
                )

        # process torso movement
        rh_neutral_range = self.calibration_data["right"]["horizontal"]["center_range"]
        rh_limits = self.calibration_data["right"]["horizontal"]["limits"]
        rv_neutral_range = self.calibration_data["right"]["vertical"]["center_range"]
        rv_limits = self.calibration_data["right"]["vertical"]["limits"]
        rh = self._rh_ema
        rv = self._rv_ema

        torso_cmd = curr_torso_q.copy()  # Must copy to avoid modifying input array
        if rh > rh_neutral_range[1] or rh < rh_neutral_range[0]:
            rh = np.clip(rh, rh_limits[0], rh_limits[1])
            delta_rh = rh - 0.5 * sum(rh_neutral_range)
            delta_rh *= -1  # because horizontal direction is reversed
            delta_rh = np.sign(delta_rh) * np.clip(
                abs(delta_rh), a_max=(rh_limits[1] - rh_limits[0]) / 2, a_min=0
            )
            torso_yaw_rotation = (
                2.0
                * delta_rh
                / (rh_limits[1] - rh_limits[0])
                * self._torso_joint_max_delta
            )
            torso_cmd[3] += torso_yaw_rotation
        if rv > rv_neutral_range[1] or rv < rv_neutral_range[0]:
            rv = np.clip(rv, rv_limits[0], rv_limits[1])
            delta_rv = rv - 0.5 * sum(rv_neutral_range)
            delta_rv = np.sign(delta_rv) * np.clip(
                abs(delta_rv), a_max=(rv_limits[1] - rv_limits[0]) / 2, a_min=0
            )
            delta_rv *= -1  # because vertical direction is reversed
            # Asymmetric speed: faster forward (joystick down), slower backward (up)
            multiplier = 1.5 if delta_rv > 0 else 2.75
            torso_cmd[2] += (
                multiplier
                * delta_rv
                / (rv_limits[1] - rv_limits[0])
                * self._torso_joint_max_delta
            )
        if self.jc_left.get_button_up():
            diff = self._torso_joints1_2_stand_q - torso_cmd[:3]
            torso_cmd[0] += np.sign(diff[0]) * self._torso_joint_max_delta * 1.74 / 2.7
            torso_cmd[1] += np.sign(diff[1]) * self._torso_joint_max_delta
            # torso_cmd[2] += np.sign(diff[2]) * self._torso_joint_max_delta * 0.96 / 2.7
        elif self.jc_left.get_button_down():
            diff = self._torso_joints1_2_squat_q - torso_cmd[:3]
            torso_cmd[0] += np.sign(diff[0]) * self._torso_joint_max_delta * 1.74 / 2.7
            torso_cmd[1] += np.sign(diff[1]) * self._torso_joint_max_delta
            # torso_cmd[2] += np.sign(diff[2]) * self._torso_joint_max_delta * 0.96 / 2.7
        torso_cmd[0] = np.clip(
            torso_cmd[0],
            self._torso_joints1_2_stand_q[0],
            self._torso_joints1_2_squat_q[0],
        )
        torso_cmd[1] = np.clip(
            torso_cmd[1],
            self._torso_joints1_2_squat_q[1],
            self._torso_joints1_2_stand_q[1],
        )
        # torso_cmd[2] = np.clip(
        #     torso_cmd[2],
        #     self._torso_joints1_2_squat_q[2],
        #     self._torso_joints1_2_stand_q[2],
        # )
        torso_cmd = np.clip(torso_cmd, self.torso_joint_low, self.torso_joint_high)

        # process gripper
        if self._gripper_toggle_mode:
            if self.jc_left.get_button_zl():
                if self._left_gripper_button_pressed_times >= 30:
                    left_gripper = (
                        1.0 if self._left_gripper_current_action < 0.5 else 0.1
                    )
                    self._left_gripper_current_action = left_gripper
                    self._left_gripper_button_pressed_times = 0
                else:
                    left_gripper = self._left_gripper_current_action
                    self._left_gripper_button_pressed_times += 1
            else:
                left_gripper = self._left_gripper_current_action

            if self.jc_right.get_button_zr():
                if self._right_gripper_button_pressed_times >= 30:
                    right_gripper = (
                        1.0 if self._right_gripper_current_action < 0.5 else 0.1
                    )
                    self._right_gripper_current_action = right_gripper
                    self._right_gripper_button_pressed_times = 0
                else:
                    right_gripper = self._right_gripper_current_action
                    self._right_gripper_button_pressed_times += 1
            else:
                right_gripper = self._right_gripper_current_action
        else:
            left_gripper = 1.0 if self.jc_left.get_button_zl() else 0.1
            right_gripper = 1.0 if self.jc_right.get_button_zr() else 0.1

        return {
            "mobile_base_cmd": np.array(base_displacement),
            "torso_controller": "joint_position",
            "torso_cmd": torso_cmd,
            "gripper_cmd": {
                "left": left_gripper,
                "right": right_gripper,
            },
        }

    def __del__(self):
        del self.jc_left
        del self.jc_right

    def close(self):
        self.__del__()
