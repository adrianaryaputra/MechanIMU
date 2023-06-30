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

def ned2lla(ned_pos, base_lla):
    ref_lat, ref_lon, ref_alt = base_lla.reshape(3,).tolist()

    # Convert reference LLA to Cartesian coordinates
    ref_x, ref_y, ref_z = lla2ecef(ref_lat, ref_lon, ref_alt)

    # Calculate LLA position difference
    diff_x = -ned_pos[1,0]  # East to West
    diff_y = ned_pos[0,0]   # North to South
    diff_z = -ned_pos[2,0]  # Down to Up

    # Convert LLA position difference to Cartesian coordinates
    new_lat, new_lon, new_alt = ecef2lla(
        ref_x + diff_x, 
        ref_y + diff_y, 
        ref_z + diff_z
    )

    return np.array([new_lat, new_lon, new_alt]).reshape(3,1)

def lla2ecef(lat, lon, alt):
    # Conversion constants
    a = 6378137.0  # Equatorial radius in meters
    f_inv = 298.257223563  # Inverse flattening
    f = 1 / f_inv

    # Calculate intermediate values
    e2 = 1 - (1 - f) * (1 - f)
    N = a / (1 - e2 * (np.sin(lat) ** 2)) ** 0.5

    # Convert LLA to Cartesian coordinates
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)

    return x, y, z

def ecef2lla(x, y, z):
    # Conversion constants
    a = 6378137.0  # Equatorial radius in meters
    f_inv = 298.257223563  # Inverse flattening
    f = 1 / f_inv

    # Calculate intermediate values
    e2 = 1 - (1 - f) * (1 - f)
    p = np.sqrt(x ** 2 + y ** 2)

    # Calculate latitude
    lat = np.arctan2(z, (p * (1 - e2)))

    # Calculate longitude
    lon = np.arctan2(y, x)

    # Calculate altitude
    N = a / np.sqrt(1 - e2 * (np.sin(lat) ** 2))
    alt = (p / np.cos(lat)) - N

    return np.array([lat, lon, alt]).reshape(3,1)