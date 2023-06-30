"""
Mechanization of the INS implementation
written by: adrianaryaputra, vincentius, aliffudinakbar
"""

import numpy as np
from src.tools import CoordinateTransform as CT
from src.mechanization import Mechanization

# if __name__ == "__main__":
#     # ce2n = createCT_earth2nav(np.radians(30), np.radians(45))
#     # print(ce2n.T.dcm)
#     # from scipy.spatial.transform import Rotation as R
#     # r = R.from_euler("zyx", [-45, -(-30-90), 0], degrees=True)
#     # print(r.as_matrix())

#     cb2n = CT.create_body2nav(np.radians(30), np.radians(45), np.radians(50))
#     print(cb2n.T.dcm)
#     from scipy.spatial.transform import Rotation as R
#     r = R.from_euler("zyx", [-50, -45, -30], degrees=True)
#     print(r.as_matrix())

if __name__ == "__main__":
    ins = Mechanization.INS([np.radians(-7.7677225), np.radians(110.3861133), 0], [0, 0, 0])
    ins.calculate_gyro(
        np.array([[0], [0], [0]]),
        np.array([[0], [0], [0]]),
        0.01
    )
    print("rLLA", ins.state.positionLLA)
    print("r", ins.state.position)
    print("v", ins.state.velocity)
    print("a", ins.state.acceleration)
    print("q", ins.state.quaternion)
    ins.calculate_accel(
        np.array([[0], [0], [0]]), 
        np.array([[0], [0], [0]]), 
        time_step=0.01, 
        sgx=0.00, sgy=0.00, sgz=0.00
    )
    print("rLLA", ins.state.positionLLA)
    print("r", ins.state.position)
    print("v", ins.state.velocity)
    print("a", ins.state.acceleration)
    print("q", ins.state.quaternion)
    ins.calculate_gyro(
        np.array([[0], [0], [0]]),
        np.array([[0], [0], [0]]),
        0.01
    )
    print("rLLA", ins.state.positionLLA)
    print("r", ins.state.position)
    print("v", ins.state.velocity)
    print("a", ins.state.acceleration)
    print("q", ins.state.quaternion)
    ins.calculate_accel(
        np.array([[0], [0], [0]]), 
        np.array([[0], [0], [0]]), 
        time_step=0.01, 
        sgx=0.00, sgy=0.00, sgz=0.00
    )
    print("rLLA", ins.state.positionLLA)
    print("r", ins.state.position)
    print("v", ins.state.velocity)
    print("a", ins.state.acceleration)
    print("q", ins.state.quaternion)
    