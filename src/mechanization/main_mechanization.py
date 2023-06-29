"""
Main mechanization module
written by: adrianaryaputra, vincentius, aliffudinakbar
"""

from dataclasses import dataclass
import numpy as np

from src.tools import CoordinateTransform as ct

class INSConstant:
    """
    Initialize the constant of the INS
    """
    earth_rotation_speed = 7.2921158
    earth_eccentricity = 0.0818191908425
    earth_equatorial_radius = 6378.137
    
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
        dcm earth2nav -- C_{e}^{n} = DCM factory
        dcm nav2body -- C_{n}^{b} = DCM factory
        dec body2nav -- C_{b}^{n} = DCM factory
    """
    position: np.ndarray = np.zeros((3,1))
    velocity: np.ndarray = np.zeros((3,1))
    acceleration: np.ndarray = np.zeros((3,1))
    gravity_gamma_n: np.ndarray = np.zeros((3,1))
    specific_force_f_b: np.ndarray = np.zeros((3,1))
    specific_force_f_n: np.ndarray = np.zeros((3,1))

    omega_e_ie: np.ndarray = np.array([
        [0.0],
        [0.0],
        [INSConstant.earth_rotation_speed]
    ])
    omega_b_ib: np.ndarray = np.zeros((3,1))
    omega_b_in: np.ndarray = np.zeros((3,1))
    omega_b_nb: np.ndarray = np.zeros((3,1))
    omega_n_en: np.ndarray = np.zeros((3,1))
    omega_n_in: np.ndarray = np.zeros((3,1))
    omega_n_ie: np.ndarray = np.zeros((3,1))

    # pylint: disable=invalid-name
    C_n2b: np.ndarray = np.zeros((3,3))
    C_b2n: np.ndarray = np.zeros((3,3))
    C_e2n: np.ndarray = np.zeros((3,3))

    # pylint: disable=missing-function-docstring
    def dcm_e_n(self, lat, long):
        return ct.create_earth2nav(lat, long)

    def dcm_n_b(self, roll, pitch, yaw):
        return ct.create_nav2body(roll, pitch, yaw)

    def dcm_b_n(self, roll, pitch, yaw):
        return ct.create_body2nav(roll, pitch, yaw)
    
    # pylint: disable=invalid-name
    def earth_M(self, latitude):
        """
        Earth's Prime Meridian
        """
        return INSConstant.earth_equatorial_radius * (1 - INSConstant.earth_eccentricity**2) / (1 - INSConstant.earth_eccentricity**2 * np.sin(latitude)**2)**(3/2)
    
    def earth_N(self, latitude):
        """
        Earth's Prime Vertical
        """
        return INSConstant.earth_equatorial_radius / (1 - INSConstant.earth_eccentricity**2 * np.sin(latitude)**2)**(1/2)

class INS:
    """
    Inertial Navigation System Mechanization
    """
    # pylint: disable=invalid-name
    def __init__(self, gps_LLA, magneto_RPY):
        # state initialization
        self.state = INSState()

    def update_feedback_LLA(self, latitude, longitude, altitude):
        # feedback update
        self.state.C_e2n = self.state.dcm_e_n(latitude, longitude)
        self.state.omega_n_en = self._update_omega_n_en(self.state, latitude, altitude)
        self.state.omega_n_ie = self.state.C_e2n @ self.state.omega_e_ie
        self.state.omega_n_in = self.state.omega_n_ie + self.state.omega_n_en
        self.state.gravity_gamma_n = self._update_gravity_gamma_n(latitude, altitude)

    def update_feedback_RPY(self, roll, pitch, yaw):
        # feedback update
        self.state.C_n2b = self.state.dcm_n_b(roll, pitch, yaw)
        self.state.C_b2n = self.state.C_n2b.T
        self.state.omega_b_in = self.state.C_n2b @ self.state.omega_n_in
    
    def calculate_gyro(self, gyro, timediff):
        # calculate gyro
        self.state.omega_b_ib = gyro
        self.state.omega_b_nb = self.state.omega_b_ib - self.state.omega_b_in
        # self.state.C_n2b += self.state.omega_b_nb * timediff

    @staticmethod
    def _update_omega_n_en(state, latitude, altitude):
        """
        Update the angular velocity of nav frame wrt earth frame
        """
        return np.array([
            [state.velocity[1,0] / (state.earth_N(latitude) + altitude)],
            [-state.velocity[0,0] / (state.earth_M(latitude) + altitude)],
            [-state.velocity[1,0] * np.tan(latitude) / (state.earth_N(latitude) + altitude)]
        ])
        
    @staticmethod
    def _update_gravity_gamma_n(latitude, altitude):
        """
        Update the gravity vector in nav frame
        """
        # pylint: disable=invalid-name
        a1 = 9.7803267715
        a2 = 0.0052790414
        a3 = 0.0000232718
        a4 = -0.0000030876910891
        a5 = 0.0000000043977311
        a6 = 0.0000000000007211
        return np.array([
            [0],
            [0],
            [a1 * (1 + a2 * np.sin(latitude)**2 + a3 * np.sin(latitude)**4) + (a4 + a5 * np.sin(latitude)**2)*altitude + a6 * altitude**2]
        ])