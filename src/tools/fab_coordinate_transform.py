"""
Factory functions for creating DCMs for coordinate transformations.
written by : adrianaryaputra, vincentius, aliffudinakbar
"""

import numpy as np
from src.dcm import DCM

def create_earth2nav(lat_radians, long_radians) -> DCM:
    """
    Creates a DCM for transforming from earth to navigation frame.
    @symbol: C_{e}^{n}

    @param lat_radians: latitude in radians
    @param long_radians: longitude in radians
    
    @return: DCM object
    """
    dcm = DCM()
    dcm.compile_rotate(pitch=-lat_radians-np.pi/2, yaw=long_radians)
    return dcm

def create_nav2earth(lat_radians, long_radians) -> DCM:
    """
    Creates a DCM for transforming from navigation to earth frame.
    @symbol: C_{n}^{e}

    @param lat_radians: latitude in radians
    @param long_radians: longitude in radians

    @return: DCM object
    """
    dcm = create_earth2nav(lat_radians, long_radians)
    return dcm.T

def create_body2nav(roll_radians, pitch_radians, yaw_radians) -> DCM:
    """
    Creates a DCM for transforming from body to navigation frame.
    @symbol: C_{b}^{n}

    @param roll_radians: roll angle in radians
    @param pitch_radians: pitch angle in radians
    @param yaw_radians: yaw angle in radians

    @return: DCM object
    """
    dcm = DCM()
    dcm.compile_rotate(roll=roll_radians, pitch=pitch_radians, yaw=yaw_radians)
    return dcm

def create_nav2body(roll_radians, pitch_radians, yaw_radians) -> DCM:
    """
    Creates a DCM for transforming from navigation to body frame.
    @symbol: C_{n}^{b}

    @param roll_radians: roll angle in radians
    @param pitch_radians: pitch angle in radians
    @param yaw_radians: yaw angle in radians

    @return: DCM object
    """
    dcm = create_body2nav(roll_radians, pitch_radians, yaw_radians)
    return dcm.T
