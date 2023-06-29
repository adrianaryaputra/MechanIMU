import numpy as np
from src.dcm import DCM

def createCT_earth2nav(latRad, longRad):
    dcm = DCM()
    dcm.compileRotate(pitch=-latRad-np.pi/2, yaw=longRad)
    return dcm

def createCT_nav2earth(latRad, longRad):
    dcm = createCT_earth2nav(latRad, longRad)
    return dcm.T

def createCT_body2nav(rollRad, pitchRad, yawRad):
    dcm = DCM()
    dcm.compileRotate(roll=rollRad, pitch=pitchRad, yaw=yawRad)
    return dcm

def createCT_nav2body(rollRad, pitchRad, yawRad):
    dcm = createCT_body2nav(rollRad, pitchRad, yawRad)
    return dcm.T


