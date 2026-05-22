import numpy as np
from scipy.integrate import odeint

from ball_estimation.trajectory import Trajectory3D
from ball_estimation.trajectory.model import (
    ArcMotionParameters,
    TrajectoryModel,
)

from ball_estimation.trajectory.optimization import ArcOptimizationBounds


class DEQMagnusArcMotionParameters(ArcMotionParameters):
    def __init__(self, x0=0., y0=0.,
                 z0=0., vx0=0., vy0=0., vz0=0.,
                 wx0=0., wy0=0., wz0=0.,
                 g=0., k3=0., kl=0., ks=0.,
                 km=0):
        super().__init__(x0, y0, z0, vx0, vy0, vz0,
                         g, k3, kl, ks)
        self.wx0 = wx0
        self.wy0 = wy0
        self.wz0 = wz0
        self.km = km


class DEQMagnusArcOptimizationBounds(ArcOptimizationBounds):
    def __init__(self, allowed_startpoint_shift_arc,
                 max_ball_speed, gravitation_g, g_variation,
                 k3, k3_variation, kl, ks, kl_ks_variation, angular_velocity_factor, km, km_variation):

        super().__init__(allowed_startpoint_shift_arc,
                 max_ball_speed, gravitation_g, g_variation,
                 k3, k3_variation, kl, ks, kl_ks_variation)
        self.angular_velocity_factor = angular_velocity_factor
        self.km = km
        self.km_variation = km_variation

    def build_bounds(self):
        self.lower_bounds = DEQMagnusArcMotionParameters()
        self.upper_bounds = DEQMagnusArcMotionParameters()

    def update(self, xh_start, yh_start):
        SMALL_EPS = 1e-5
        DM = 1 / 10
        super().update(xh_start, yh_start)
        ball_radius = DEQMagnusTrajectoryModel.BALL_RADIUS
        scale = self.angular_velocity_factor / (ball_radius / DM)
        self.lower_bounds.wx0 = self.lower_bounds.vx0 * scale - SMALL_EPS
        self.upper_bounds.wx0 = self.upper_bounds.vx0 * scale + SMALL_EPS
        self.lower_bounds.wy0 = self.lower_bounds.vy0 * scale - SMALL_EPS
        self.upper_bounds.wy0 = self.upper_bounds.vy0 * scale + SMALL_EPS
        self.lower_bounds.wz0 = self.lower_bounds.vz0 * scale - SMALL_EPS
        self.upper_bounds.wz0 = self.upper_bounds.vz0 * scale + SMALL_EPS
        self.lower_bounds.km = self.lower_bounds.km - SMALL_EPS
        self.upper_bounds.km = self.upper_bounds.km + SMALL_EPS


class DEQMagnusTrajectoryModel(TrajectoryModel):
    BALL_RADIUS = 0.11
    BALL_MASS = 0.43
    motion_parameters_type = DEQMagnusArcMotionParameters

    def __init__(self):
        pass

    def simulate(self, initial_trajectory: Trajectory3D, motion_params):
        dt = initial_trajectory.time[1] - initial_trajectory.time[0]
        x, y, z = self.get_arc_trajectory(**motion_params.__dict__,
                                               num_points=len(initial_trajectory.time),
                                            fps=1/dt,
                                         )
        out_trajectory = Trajectory3D.from_xyz(x, y, z, initial_trajectory.time)
        return out_trajectory
    @staticmethod
    def get_arc_trajectory(
        x0, y0, z0, vx0, vy0, vz0, k3, num_points, fps, g, kl=0, ks=0, wx0=0, wy0=0, wz0=0, km=0.0
    ):
        """Gives the ball parabolic trajectory when moving in the air

        Parameters
        ----------
        x0, y0, z0: float
            X,Y,Z coordinates of ball starting point
        vx0, vy0, vz0: float
            starting velocity X,Y,Z coordinates
        k3: float
            air resistance coefficient
        num_points: int
            number of points required in trajectory
        fps: int
            video frames per second
        g: float
            gravitational constant (normally 9.8)
        kl: float
            top-spin coefficient
        ks: float
            side-spin coefficient
        wx0, wy0, wz0: float
            starting angular velocity X,Y,Z coordinates
        km: float
            magnus coefficient

        Returns
        -------
        xt, yt, zt: list of float
            lists of trajectory points coordinates
        """

        def dU_dt(U, t):
            # Here U is a vector such that X=U[0], Y=U[1], Z=U[2], Vx=U[3], Vy=U[4], Vz=U[5],
            # Wx=U[6], Wy=U[7], Wz=U[8].

            # Extract the current state values from the vector U
            X, Y, Z = U[0], U[1], U[2]
            Vx, Vy, Vz = U[3], U[4], U[5]
            Wx, Wy, Wz = U[6], U[7], U[8]

            # Calculate the magnitude of the velocity
            V = np.sqrt(Vx ** 2 + Vy ** 2 + Vz ** 2)

            # Compute the Magnus force components
            Magnus_x = km * (Vz * Wy - Vy * Wz)
            Magnus_y = km * (Vx * Wz - Vz * Wx)
            Magnus_z = km * (Vy * Wx - Vx * Wy)

            # Drag forces acting on the ball
            drag_x = -k3 * V * Vx
            drag_y = -k3 * V * Vy
            drag_z = -k3 * V * Vz

            # Gravity force
            gravity = -g

            # Linear velocity derivatives (same as before)
            dX_dt = Vx
            dY_dt = Vy
            dZ_dt = Vz

            # Linear acceleration components (including drag and Magnus effect)
            dVx_dt = drag_x + Magnus_x
            dVy_dt = drag_y + Magnus_y
            dVz_dt = gravity + drag_z + Magnus_z

            # Moment of inertia for a sphere (I = 2/5 * m * r^2), assuming uniform density
            I = (2 / 5) * DEQMagnusTrajectoryModel.BALL_MASS * DEQMagnusTrajectoryModel.BALL_RADIUS ** 2  # m = mass, r = radius of the ball

            # Compute the torques (Magnus effect only, for simplicity here)
            # These torques cause changes in angular velocity
            torque_x = Magnus_x * DEQMagnusTrajectoryModel.BALL_RADIUS  # Effective torque due to Magnus force
            torque_y = Magnus_y * DEQMagnusTrajectoryModel.BALL_RADIUS
            torque_z = Magnus_z * DEQMagnusTrajectoryModel.BALL_RADIUS

            # Angular velocity derivatives (assuming the torques from Magnus are responsible)
            dWx_dt = torque_x / I
            dWy_dt = torque_y / I
            dWz_dt = torque_z / I

            # Return the derivatives in the same order as the state vector U
            return [dX_dt, dY_dt, dZ_dt, dVx_dt, dVy_dt, dVz_dt, dWx_dt, dWy_dt, dWz_dt]

        U0 = [
            x0 / 10,
            y0 / 10,
            z0 / 10,
            vx0 / 10,
            vy0 / 10,
            vz0 / 10,
            wx0,
            wy0,
            wz0,
        ]  # decimeters to meters
        t = (1 / fps) * np.linspace(0, num_points - 1, num_points)
        Us = odeint(dU_dt, U0, t)

        xt = list(10 * Us[:, 0])  # meters to decimeters
        yt = list(10 * Us[:, 1])
        zt = list(10 * Us[:, 2])

        return xt, yt, zt

