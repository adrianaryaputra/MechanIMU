"""
This module contains mathematical tools for the INS implementation.
written by: adrianaryaputra, vincentius, aliffudinakbar
"""

import numpy as np

def skew_symmetric_matrix(vector, diag=0, scaler=1):
    """
    Returns the skew-symmetric matrix of a vector.
    """
    arr = np.zeros((3,3))
    arr[0,0] = diag
    arr[1,0] = vector[2,0]*scaler
    arr[2,0] = -vector[1,0]*scaler
    
    arr[0,1] = -vector[2,0]*scaler
    arr[1,1] = diag
    arr[2,1] = vector[0,0]*scaler
    
    arr[0,2] = vector[1,0]*scaler
    arr[1,2] = -vector[0,0]*scaler
    arr[2,2] = diag

    return arr
    # return np.array([
    #     [diag, -vector[2]*scaler, vector[1]*scaler],
    #     [vector[2]*scaler, diag, -vector[0]*scaler],
    #     [-vector[1]*scaler, vector[0]*scaler, diag]
    # ])

def ecef_to_lla(x, y, z):
    # Constants
    a = 6378137  # Semi-major axis of the Earth
    e = 0.0818191908426  # First Eccentricity of the Earth

    # Calculations
    b = np.sqrt(a**2 * (1 - e**2))  # Semi-minor axis
    ep = np.sqrt((a**2 - b**2) / b**2)  # Second Eccentricity
    p = np.sqrt(x**2 + y**2)  # Distance from spin axis
    th = np.arctan2(a * z, b * p)  # Angle related with true latitude
    lon = np.arctan2(y, x)  # Longitude
    lat = np.arctan2((z + ep**2 * b * np.sin(th)**3), (p - e**2 * a * np.cos(th)**3))  # Latitude
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)  # Radius of curvature in the prime vertical
    alt = p / np.cos(lat) - N  # Altitude

    # Convert to degrees
    lon = np.degrees(lon)
    lat = np.degrees(lat)

    return np.array([lat, lon, alt]).reshape((3,1))