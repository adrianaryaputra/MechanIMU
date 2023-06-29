"""
Mechanization of the INS implementation
written by: adrianaryaputra, vincentius, aliffudinakbar
"""

import numpy as np
from src.tools import CoordinateTransform as CT

if __name__ == "__main__":
    # ce2n = createCT_earth2nav(np.radians(30), np.radians(45))
    # print(ce2n.T.dcm)
    # from scipy.spatial.transform import Rotation as R
    # r = R.from_euler("zyx", [-45, -(-30-90), 0], degrees=True)
    # print(r.as_matrix())

    cb2n = CT.create_body2nav(np.radians(30), np.radians(45), np.radians(50))
    print(cb2n.T.dcm)
    from scipy.spatial.transform import Rotation as R
    r = R.from_euler("zyx", [-50, -45, -30], degrees=True)
    print(r.as_matrix())
    