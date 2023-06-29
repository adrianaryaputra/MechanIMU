"""
Main mechanization module
written by: adrianaryaputra, vincentiuscharles, aliffudinakbar
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.tools import CoordinateTransform as ct
from src.tools import MathKit as mk

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
    positionLLA: np.ndarray = np.zeros((3,1))
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

    quaternion: np.ndarray = np.zeros((4,1))

    # pylint: disable=invalid-name
    C_n2b: np.ndarray = np.zeros((3,3))
    C_b2n: np.ndarray = np.zeros((3,3))
    C_e2n: np.ndarray = np.zeros((3,3))
    C_n2e: np.ndarray = np.zeros((3,3))

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
        self.update_feedback_LLA(gps_LLA[0], gps_LLA[1], gps_LLA[2])
        self.update_feedback_RPY(magneto_RPY[0], magneto_RPY[1], magneto_RPY[2])

    def update_feedback_LLA(self, latitude, longitude, altitude):
        # feedback update
        self.state.positionLLA = np.array([
            [latitude],
            [longitude],
            [altitude]
        ])
        self.state.C_e2n = self.state.dcm_e_n(latitude, longitude)
        self.state.omega_n_en = self._update_omega_n_en(self.state, latitude, altitude)
        self.state.omega_n_ie = self.state.C_e2n @ self.state.omega_e_ie
        self.state.omega_n_in = self.state.omega_n_ie + self.state.omega_n_en
        self.state.gravity_gamma_n = self._update_gravity_gamma_n(latitude, altitude)

    def update_feedback_RPY(self, roll, pitch, yaw):
        # feedback update
        self.state.C_n2b = self.state.dcm_n_b(roll, pitch, yaw)
        self.state.C_b2n = self.state.C_n2b.T
        self.state.quaternion = ct.DCM.to_quaternion(self.state.C_b2n).reshape(4,1)
        self.state.omega_b_in = self.state.C_n2b @ self.state.omega_n_in
    
    def calculate_gyro(self, gyro, bias_gyro, time_step):
        # calculate gyro
        self.state.omega_b_ib = gyro - (bias_gyro * time_step) # sama dengan delta_theta_b_ib
        self.state.omega_b_nb = self.state.omega_b_ib - (self.state.omega_b_in * time_step)
        delta_theta = np.sqrt(self.state.omega_b_nb[0,0]**2 + self.state.omega_b_nb[1,0]**2 + self.state.omega_b_nb[2,0]**2)
        hs = (2/delta_theta) * np.sin(delta_theta/2)
        hc = 2 * (np.cos(delta_theta/2) - 1)
        print(self.state.quaternion)
        self.state.quaternion += 0.5 * np.matrix([
            [hc, hs*self.state.omega_b_nb[2,0], -hs*self.state.omega_b_nb[1,0], hs*self.state.omega_b_nb[0,0]],
            [-hs*self.state.omega_b_nb[2,0], hc, hs*self.state.omega_b_nb[0,0], hs*self.state.omega_b_nb[1,0]],
            [hs*self.state.omega_b_nb[1,0], -hs*self.state.omega_b_nb[0,0], hc, hs*self.state.omega_b_nb[2,0]],
            [-hs*self.state.omega_b_nb[0,0], -hs*self.state.omega_b_nb[1,0], -hs*self.state.omega_b_nb[2,0], hc]
        ]) @ self.state.quaternion

        # convert quaternion to euler angle
        attitude_rpy = R.from_quat(self.state.quaternion.reshape(4,)).as_euler("xyz")
        self.update_feedback_RPY(attitude_rpy[0], attitude_rpy[1], attitude_rpy[2])


    def calculate_accel(self, accel, bias_accel, time_step, **accel_cfg):
        # calculate accel
        print("accel_cfg", accel_cfg)
        sg = np.diag([1/(1+accel_cfg['sgx']), 1/(1+accel_cfg['sgy']), 1/(1+accel_cfg['sgz'])])
        self.state.specific_force_f_b = sg @ (accel - (bias_accel * time_step))
        sks = mk.skew_symmetric_matrix(self.state.omega_b_nb, diag=1, scaler=0.5)
        self.state.specific_force_f_n = self.state.C_b2n @ sks @ self.state.specific_force_f_b
        w_n_in = (2*self.state.omega_n_ie + self.state.omega_n_en)
        acceleration_fb = np.cross(w_n_in.reshape((3,)), self.state.velocity.reshape((3,))) * time_step
        self.state.acceleration = self.state.specific_force_f_n - acceleration_fb + (self.state.gravity_gamma_n * time_step)

        last_velocity = self.state.velocity.copy()
        self.state.velocity = self.state.velocity + self.state.acceleration

        d_one = np.matrix([
            [1/self.state.earth_M(self.state.positionLLA[0,0]) + self.state.positionLLA[2,0], 0, 0],
            [0, 1/(self.state.earth_N(self.state.positionLLA[0,0]) + self.state.positionLLA[2,0])*np.cos(self.state.positionLLA[0,0]), 0],
            [0, 0, -1]
        ])

        self.state.position = self.state.position + (0.5 * d_one) @ (self.state.velocity + last_velocity) * time_step

        # update positionLLA
        # converting from navigation frame to earth frame
        self.state.C_n2e = ct.create_nav2earth(self.state.positionLLA[0,0], self.state.positionLLA[1,0])
        positionECEF = self.state.C_n2e @ self.state.position
        # converting from earth frame to LLA
        self.state.positionLLA = mk.ecef_to_lla(positionECEF[0,0], positionECEF[1,0], positionECEF[2,0])

        # do LLA update
        self.update_feedback_LLA(self.state.positionLLA[0,0], self.state.positionLLA[1,0], self.state.positionLLA[2,0])

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


if __name__ == "__main__":
    ins = INS([np.radians(-7.7677225), np.radians(110.3861133), 0], [0, 0, 0])
    ins.calculate_gyro(
        np.array([[0], [0], [0]]),
        np.array([[0], [0], [0]]),
        0.01
    )
    print(ins.state.positionLLA)
    print(ins.state.position)
    print(ins.state.velocity)
    print(ins.state.acceleration)
    print(ins.state.quaternion)
    ins.calculate_accel(
        np.array([[0], [0], [9.8]]), 
        np.array([[0], [0], [0]]), 
        0.01, 
        sgx=0.01, sgy=0.01, sgz=0.01
    )
    print(ins.state.positionLLA)
    print(ins.state.position)
    print(ins.state.velocity)
    print(ins.state.acceleration)
    print(ins.state.quaternion)