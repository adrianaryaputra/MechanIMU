import numpy as np

class DCM:
    def __init__(self):
        self.dcm = np.identity(3)
        self.rotationHistory = []
        self.initiated = False

    def __matmul__(self, other):
        if not self.initiated:
            raise RuntimeError("DCM is not initiated yet")
        if not isinstance(other, np.ndarray):
            raise TypeError("DCM can only be multiplied with numpy.ndarray<3> or numpy.ndarray<3,1>")
        if other.shape in [(3,), (3, 1)]:
            return np.matmul(self.dcm, other)
        raise TypeError("DCM can only be multiplied with numpy.ndarray<3> or numpy.ndarray<3,1>")

    def __array__(self):
        if not self.initiated:
            raise RuntimeError("DCM is not initiated yet")
        return self.dcm

    def compileRotate(self, roll=None, pitch=None, yaw=None):
        if roll is not None:
            self.dcm = self._rotateX(roll) @ self.dcm
            self.rotationHistory.append({"axis": "x", "angle": roll})
        if pitch is not None:
            self.dcm = self._rotateY(pitch) @ self.dcm
            self.rotationHistory.append({"axis": "y", "angle": pitch})
        if yaw is not None:
            self.dcm = self._rotateZ(yaw) @ self.dcm
            self.rotationHistory.append({"axis": "z", "angle": yaw})
        self.initiated = True

    @property
    def T(self):
        if not self.initiated:
            raise RuntimeError("DCM is not initiated yet")
        tempDCM = self.dcm.copy()
        tempDCM = tempDCM.T
        tempHist = self.rotationHistory.copy()
        tempHist.reverse()
        for i in range(len(tempHist)):
            tempHist[i]["angle"] *= -1
        return DCM.fromArrayHistory(tempDCM, tempHist)

    @staticmethod
    def fromArrayHistory(arr, history):
        if not isinstance(arr, np.ndarray):
            raise TypeError("arr must be a numpy.ndarray<3,3>")
        if not isinstance(history, list):
            raise TypeError("history must be a list of dict")
        #if len(history) != 3:
        #    raise ValueError("history must be a list of dict with length of 3")
        for i in range(len(history)):
            if not isinstance(history[i], dict):
                raise TypeError("history must be a list of dict")
            if "axis" not in history[i] or "angle" not in history[i]:
                raise ValueError("history must be a list of dict with keys 'axis' and 'angle'")
            if history[i]["axis"] not in ["x", "y", "z"]:
                raise ValueError("history must be a list of dict with 'axis' value of 'x', 'y', or 'z'")
            if not isinstance(history[i]["angle"], (int, float)):
                raise TypeError("history must be a list of dict with 'angle' value of int or float")
        dcm = DCM()
        dcm.dcm = arr
        dcm.rotationHistory = history
        dcm.initiated = True
        return dcm

    @staticmethod
    def _rotateX(roll):
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        return np.array([
            [1, 0, 0], 
            [0, cos_roll, -sin_roll], 
            [0, sin_roll, cos_roll]
        ])

    @staticmethod
    def _rotateY(pitch):
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        return np.array([
            [cos_pitch, 0, sin_pitch], 
            [0, 1, 0], 
            [-sin_pitch, 0, cos_pitch]
        ])

    @staticmethod
    def _rotateZ(yaw):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return np.array([
            [cos_yaw, -sin_yaw, 0], 
            [sin_yaw, cos_yaw, 0], 
            [0, 0, 1]
        ])


if __name__ == "__main__":
    # do some testing here
    # 1. using DCM class to calculate the rotation matrix
    dcm = DCM()
    dcm.compileRotate(roll=np.radians(-20), pitch=np.radians(-30), yaw=np.radians(-40))
    print("DCM", dcm.dcm)
    print("DCM.T", dcm.T.dcm)
    print("History", dcm.rotationHistory)
    print("History.T", dcm.T.rotationHistory)
    # 2. using scipy rotation to calculate the rotation matrix
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler("xyz", [-20, -30, -40], degrees=True)
    print("Scipy", r.as_matrix())
    rrt = R.from_euler("zyx", [40, 30, 20], degrees=True)
    print("Scipy.T", rrt.as_matrix())