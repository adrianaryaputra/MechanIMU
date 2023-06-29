"""
written by: adrianaryaputra, vincentius, aliffudinakbar
"""

import numpy as np

class DCM:
    """
    Direction Cosine Matrix (DCM) class
    This class is used to calculate the rotation matrix from a series of rotation
    """
    def __init__(self):
        self.dcm = np.identity(3)
        self._history = []
        self._initiated = False

    def __matmul__(self, other):
        if not self._initiated:
            raise RuntimeError("DCM is not initiated yet")
        if not isinstance(other, np.ndarray):
            raise TypeError("DCM can only be multiplied with numpy.ndarray<3> or numpy.ndarray<3,1>")
        if other.shape in [(3,3), (3, 2), (3,1)]:
            return np.matmul(self.dcm, other)
        raise TypeError("DCM can only be multiplied with numpy.ndarray<3> or numpy.ndarray<3,1>")

    def __array__(self):
        if not self._initiated:
            raise RuntimeError("DCM is not initiated yet")
        return self.dcm

    def compile_rotate(self, roll=None, pitch=None, yaw=None):
        """
        Compile the rotation matrix from a series of rotation
        @param roll: roll angle in radians
        @param pitch: pitch angle in radians
        @param yaw: yaw angle in radians

        @return: None
        """
        if roll is not None:
            self.dcm = self._rotate_x(roll) @ self.dcm
            self._history.append({"axis": "x", "angle": roll})
        if pitch is not None:
            self.dcm = self._rotate_y(pitch) @ self.dcm
            self._history.append({"axis": "y", "angle": pitch})
        if yaw is not None:
            self.dcm = self._rotate_z(yaw) @ self.dcm
            self._history.append({"axis": "z", "angle": yaw})
        self._initiated = True

    @property
    # disable pylint-invalid-name because it is a property
    # pylint: disable=invalid-name
    def T(self):
        """
        Returns the transpose of the DCM

        @return: DCM object
        """
        if not self._initiated:
            raise RuntimeError("DCM is not initiated yet")
        temp_dcm = self.dcm.copy()
        temp_dcm = temp_dcm.T
        temp_history = self._history.copy()
        temp_history.reverse()
        for i,_ in enumerate(temp_history):
            temp_history[i]["angle"] *= -1
        return DCM.from_array_history(temp_dcm, temp_history)

    @property
    # disable pylint-invalid-name because it is a property
    # pylint: disable=invalid-name
    def history(self):
        """
        Returns the history of the DCM

        @return: list of dict with keys "axis" and "angle"
        """
        if not self._initiated:
            raise RuntimeError("DCM is not initiated yet")
        return self._history

    @property
    # disable pylint-invalid-name because it is a property
    # pylint: disable=invalid-name
    def initiated(self):
        """
        Returns the initiation status of the DCM

        @return: bool
        """
        return self._initiated
    
    @property
    # disable pylint-invalid-name because it is a property
    # pylint: disable=invalid-name
    def quat(self):
        """
        Returns the quaternion representation of the DCM

        @return: numpy.ndarray<4>
        """
        return DCM.to_quaternion(self)

    @staticmethod
    def from_array_history(arr, history):
        """
        Create a DCM object from an array and history

        @param arr: numpy.ndarray<3,3>
        @param history: list of dict with keys "axis" and "angle"

        @return: DCM object
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError("arr must be a numpy.ndarray<3,3>")
        if not isinstance(history, list):
            raise TypeError("history must be a list of dict")
        #if len(history) != 3:
        #    raise ValueError("history must be a list of dict with length of 3")
        for i,_ in enumerate(history):
            if not isinstance(history[i], dict):
                raise TypeError("history must be a list of dict")
            if "axis" not in history[i] or "angle" not in history[i]:
                raise ValueError("history must be a list of dict with keys 'axis' and 'angle'")
            if history[i]["axis"] not in ["x", "y", "z"]:
                raise ValueError("history must be a list of dict with 'axis' value of 'x', 'y', or 'z'")
            if not isinstance(history[i]["angle"], (int, float)):
                raise TypeError("history must be a list of dict with 'angle' value of int or float")
        temp_dcm = DCM()
        temp_dcm.dcm = arr
        # pylint: disable=protected-access
        temp_dcm._history = history
        temp_dcm._initiated = True
        return temp_dcm
    
    @staticmethod
    def from_quaternion(quaternion):
        """
        Create a DCM object from a quaternion
        """
        if not isinstance(quaternion, np.ndarray):
            raise TypeError("quaternion must be a numpy.ndarray<4>")
        if quaternion.shape != (4,):
            raise ValueError("quaternion must be a numpy.ndarray<4>")
        quat_i = quaternion[0]
        quat_j = quaternion[1]
        quat_k = quaternion[2]
        quat_s = quaternion[3]
        temp_dcm = np.array([
            [1 - 2 * (quat_j**2 + quat_k**2), 2 * (quat_i * quat_j - quat_k * quat_s), 2 * (quat_i * quat_k + quat_j * quat_s)],
            [2 * (quat_i * quat_j + quat_k * quat_s), 1 - 2 * (quat_i**2 + quat_k**2), 2 * (quat_j * quat_k - quat_i * quat_s)],
            [2 * (quat_i * quat_k - quat_j * quat_s), 2 * (quat_j * quat_k + quat_i * quat_s), 1 - 2 * (quat_i**2 + quat_j**2)]
        ])
        return DCM.from_array_history(
            arr=temp_dcm,
            history=[
                {
                    "axis": "x", 
                    "angle": np.arctan2(temp_dcm[1, 2], temp_dcm[2, 2])
                },
                {
                    "axis": "y", 
                    "angle": -np.arcsin(temp_dcm[0, 2])
                },
                {
                    "axis": "z", 
                    "angle": np.arctan2(temp_dcm[0, 1], temp_dcm[0, 0])
                }
            ]
        )

    @staticmethod
    def _rotate_x(roll):
        """
        Returns the rotation matrix for rotation around x-axis given the roll angle
        """
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        return np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])

    @staticmethod
    def _rotate_y(pitch):
        """
        Returns the rotation matrix for rotation around y-axis given the pitch angle
        """
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        return np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])

    @staticmethod
    def _rotate_z(yaw):
        """
        Returns the rotation matrix for rotation around z-axis given the yaw angle
        """
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return np.array([
            [cos_yaw, -sin_yaw, 0], 
            [sin_yaw, cos_yaw, 0], 
            [0, 0, 1]
        ])

    @staticmethod
    def to_quaternion(dcm_obj):
        """
        Returns the quaternion representation of a DCM
        """
        if not isinstance(dcm_obj, DCM):
            raise TypeError("dcm must be a DCM object")
        if not dcm_obj.initiated:
            raise RuntimeError("DCM is not initiated yet")
        quat_s = 0.5 * np.sqrt(1 + np.trace(dcm_obj.dcm))
        quat_i = 0.25 * (dcm_obj.dcm[2, 1] - dcm_obj.dcm[1, 2]) / quat_s
        quat_j = 0.25 * (dcm_obj.dcm[0, 2] - dcm_obj.dcm[2, 0]) / quat_s
        quat_k = 0.25 * (dcm_obj.dcm[1, 0] - dcm_obj.dcm[0, 1]) / quat_s
        # quat_s is the scalar part, taruh di belakang ngikut ehshin ucalgary
        return np.array([quat_i, quat_j, quat_k, quat_s])


if __name__ == "__main__":
    # do some testing here
    # 1. using DCM class to calculate the rotation matrix
    dcm = DCM()
    dcm.compile_rotate(roll=np.radians(-20), pitch=np.radians(-30), yaw=np.radians(-40))
    print("DCM", dcm.dcm)
    print("DCM.T", dcm.T.dcm)
    print("History", dcm.history)
    print("History.T", dcm.T.history)
    # 2. using scipy rotation to calculate the rotation matrix
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler("xyz", [-20, -30, -40], degrees=True)
    print("Scipy", r.as_matrix())
    rrt = R.from_euler("zyx", [40, 30, 20], degrees=True)
    print("Scipy.T", rrt.as_matrix())
    # 3. using DCM class to calculate the quaternion
    print("Quaternion", dcm.quat)
    # 4. using scipy rotation to calculate the quaternion
    print("Scipy.Quaternion", r.as_quat())
    # 5. using DCM class to create DCM from quaternion
    dcm2 = DCM.from_quaternion(dcm.quat)
    print("DCM2", dcm2.dcm)
    print("DCM", dcm.dcm)
