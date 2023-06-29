"""
Main mechanization module
written by: adrianaryaputra, vincentius, aliffudinakbar
"""

from dataclasses import dataclass
import numpy as np

from src.tools import CoordinateTransform as CT

@dataclass
class INSState:
    """
    Initialize the state of the INS
    state:
        position -- r^{n}) = Vector3
        velocity -- v^{n} = Vector3
        acceleration -- \\dot{v^{n}} = Vector3
        normal gravity at current position -- \\gamma^{n} = Vector3
        specific force <body frame> -- f^{b} = Vector3
        specific force <nav frame> -- f^{n} = Vector3
        rotvec of earth rotation rate -- \\omega^{e}_{ie} = Vector3
        rotvec of nav frame wrt earth frame -- \\omega^{n}_{en} = Vector3
        rotvec of gyro sensor -- \\omega^{b}_{ib} = Vector3
        rotvec of gyro sensor <nav frame> -- \\omega^{b}_{nb} = Vector3
        rotvec of nav frame wrt inertial frame -- \\omega^{n}_{in} = Vector3
        rotvec wnie -- \\omega^{n}_{ie} = Vector3
        rotvec wbin -- \\omega^{b}_{in} = Vector3
        dcm earth2nav -- C^{n}_{e} = Matrix3x3
        dcm nav2body -- C^{b}_{n} = Matrix3x3
        dec body2nav -- C^{n}_{b} = Matrix3x3
    """
    position: np.ndarray([0, 0, 0])
    velocity: np.ndarray([0, 0, 0])

class INS:
    """
    Inertial Navigation System Mechanization
    """
    def __init__(self):
        pass

    def _state_initialization(self):
        self.state = {}
        