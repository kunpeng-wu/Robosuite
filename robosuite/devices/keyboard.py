"""
Driver class for Keyboard controller.
"""

import numpy as np
from pynput.keyboard import Controller, Key, Listener

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class Keyboard(Device):
    """
    A minimalistic driver class for a Keyboard.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("r-f", "move arm vertically")
        print_command("z-x", "rotate arm about x-axis")
        print_command("t-g", "rotate arm about y-axis")
        print_command("c-v", "rotate arm about z-axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
        )

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for moving position
            if key.char == "w":
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key.char == "s":
                self.pos[0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key.char == "a":
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key.char == "d":
                self.pos[1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "f":
                self.pos[2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == "r":
                self.pos[2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving orientation
            elif key.char == "z":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
            elif key.char == "x":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] += 0.1 * self.rot_sensitivity
            elif key.char == "t":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] += 0.1 * self.rot_sensitivity
            elif key.char == "g":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
            elif key.char == "c":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            elif key.char == "v":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for grasping
            if key == Key.space:
                self.grasp = not self.grasp  # toggle gripper

            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()

        except AttributeError as e:
            pass


class KeyboardSelf(Device):
    """
    A minimalistic driver class for a Keyboard.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("r-f", "move arm vertically")
        print_command("z-x", "rotate arm about x-axis")
        print_command("t-g", "rotate arm about y-axis")
        print_command("c-v", "rotate arm about z-axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
        )

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for moving position
            if key.char == "w":
                self.pos[0] += self._pos_step * self.pos_sensitivity  # dec x
            elif key.char == "s":
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # inc x
            elif key.char == "a":
                self.pos[1] += self._pos_step * self.pos_sensitivity  # dec y
            elif key.char == "d":
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "f":
                self.pos[2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == "r":
                self.pos[2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving orientation
            elif key.char == "z":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
            elif key.char == "x":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] += 0.1 * self.rot_sensitivity
            elif key.char == "t":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] += 0.1 * self.rot_sensitivity
            elif key.char == "g":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
            elif key.char == "c":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            elif key.char == "v":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for grasping
            if key == Key.space:
                self.grasp = not self.grasp  # toggle gripper

            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()

        except AttributeError as e:
            pass


class KeyboardBimanual(Device):
    """
    A minimalistic driver class for a Keyboard.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=1.0, rot_sensitivity=1.0):

        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 0.05

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("shift_l", "toggle left gripper (open/close)")
        print_command("w-a-s-d", "move left arm horizontally in x-y plane")
        print_command("r-f", "move left arm vertically")
        print_command("z-x", "rotate left arm about x-axis")
        print_command("t-g", "rotate left arm about y-axis")
        print_command("c-v", "rotate left arm about z-axis")

        print_command("shift_r", "toggle right gripper (open/close)")
        print_command("i-j-k-l", "move right arm horizontally in x-y plane")
        print_command("y-h", "move right arm vertically")
        print_command("b-n", "rotate right arm about x-axis")
        print_command("m-,", "rotate right arm about y-axis")
        print_command("u-o", "rotate right arm about z-axis")

        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = {}
        self.raw_drotation = {}
        self.last_drotation = {}
        self.pos = {}
        self.last_pos = {}
        self.grasp = {}
        for arm in ["right", "left"]:
            self.rotation[arm] = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
            self.raw_drotation[arm] = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
            self.last_drotation[arm] = np.zeros(3)
            self.pos[arm] = np.zeros(3)  # (x, y, z)
            self.last_pos[arm] = np.zeros(3)
            self.grasp[arm] = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = {}
        raw_drotation = {}
        grasp = {}
        for arm in ["right", "left"]:
            dpos[arm] = self.pos[arm] - self.last_pos[arm]
            self.last_pos[arm] = np.array(self.pos[arm])
            raw_drotation[arm] = (
                self.raw_drotation[arm] - self.last_drotation[arm]
            )  # create local variable to return, then reset internal drotation
            self.last_drotation[arm] = np.array(self.raw_drotation[arm])
            grasp[arm] = int(self.grasp[arm])
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=grasp,
            reset=self._reset_state,
        )

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for moving position of left hand
            if key.char == "w":
                self.pos["left"][0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key.char == "s":
                self.pos["left"][0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key.char == "a":
                self.pos["left"][1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key.char == "d":
                self.pos["left"][1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "f":
                self.pos["left"][2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == "r":
                self.pos["left"][2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving position of right hand
            elif key.char == "i":
                self.pos["right"][0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key.char == "k":
                self.pos["right"][0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key.char == "j":
                self.pos["right"][1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key.char == "l":
                self.pos["right"][1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "h":
                self.pos["right"][2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key.char == "y":
                self.pos["right"][2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving orientation of left hand
            elif key.char == "z":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation["left"] = self.rotation["left"].dot(drot)  # rotates x
                self.raw_drotation["left"][1] -= 0.1 * self.rot_sensitivity
            elif key.char == "x":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation["left"] = self.rotation["left"].dot(drot)  # rotates x
                self.raw_drotation["left"][1] += 0.1 * self.rot_sensitivity
            elif key.char == "t":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation["left"] = self.rotation["left"].dot(drot)  # rotates y
                self.raw_drotation["left"][0] += 0.1 * self.rot_sensitivity
            elif key.char == "g":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation["left"] = self.rotation["left"].dot(drot)  # rotates y
                self.raw_drotation["left"][0] -= 0.1 * self.rot_sensitivity
            elif key.char == "c":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation["left"] = self.rotation["left"].dot(drot)  # rotates z
                self.raw_drotation["left"][2] += 0.1 * self.rot_sensitivity
            elif key.char == "v":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation["left"] = self.rotation["left"].dot(drot)  # rotates z
                self.raw_drotation["left"][2] -= 0.1 * self.rot_sensitivity

            # controls for moving orientation of right hand
            elif key.char == "b":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation["right"] = self.rotation["right"].dot(drot)  # rotates x
                self.raw_drotation["right"][1] -= 0.1 * self.rot_sensitivity
            elif key.char == "n":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation["right"] = self.rotation["right"].dot(drot)  # rotates x
                self.raw_drotation["right"][1] += 0.1 * self.rot_sensitivity
            elif key.char == "u":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation["right"] = self.rotation["right"].dot(drot)  # rotates y
                self.raw_drotation["right"][0] += 0.1 * self.rot_sensitivity
            elif key.char == "o":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation["right"] = self.rotation["right"].dot(drot)  # rotates y
                self.raw_drotation["right"][0] -= 0.1 * self.rot_sensitivity
            elif key.char == "m":
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation["right"] = self.rotation["right"].dot(drot)  # rotates z
                self.raw_drotation["right"][2] += 0.1 * self.rot_sensitivity
            elif key.char == ",":
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation["right"] = self.rotation["right"].dot(drot)  # rotates z
                self.raw_drotation["right"][2] -= 0.1 * self.rot_sensitivity


        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for grasping
            if key == Key.shift_l:
                self.grasp["left"] = not self.grasp["left"]  # toggle gripper

            elif key == Key.shift_r:
                self.grasp["right"] = not self.grasp["right"]  # toggle gripper

            # user-commanded reset
            elif key.char == "q":
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()

        except AttributeError as e:
            pass