import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET

from ball_estimation.trajectory import Trajectory3D
from ball_estimation.trajectory.model import (
    TrajectoryModel,
    ArcMotionParameters,
)

from ball_estimation.trajectory.optimization import ArcOptimizationBounds


class MujocoArcMotionParameters(ArcMotionParameters):
    def __init__(self, x0=0., y0=0.,
                 z0=0., vx0=0., vy0=0., vz0=0.,
                 wx0=0., wy0=0., wz0=0.,
                 g=0., k3=0., kl=0., ks=0.,
                 blunt_drag=0., slender_drag=0., angular_drag=0., kutta_lift=0., magnus_lift=0.):
        super().__init__(x0, y0, z0, vx0, vy0, vz0,
                         g, k3, kl, ks)
        self.wx0 = wx0
        self.wy0 = wy0
        self.wz0 = wz0
        self.blunt_drag = blunt_drag
        self.slender_drag = slender_drag
        self.angular_drag = angular_drag
        self.kutta_lift = kutta_lift
        self.magnus_lift = magnus_lift


class MujocoArcOptimizationBounds(ArcOptimizationBounds):
    DEFAULT_BLUNT = 0.5
    DEFAULT_SLENDER = 0.25
    DEFAULT_ANGULAR = 1.5
    DEFAULT_KUTTA = 1.0
    DEFAULT_MAGNUS = 0.3

    def __init__(self, allowed_startpoint_shift_arc,
                 max_ball_speed, gravitation_g, g_variation,
                 k3, k3_variation, kl, ks, kl_ks_variation, angular_velocity_factor):

        super().__init__(allowed_startpoint_shift_arc,
                 max_ball_speed, gravitation_g, g_variation,
                 k3, k3_variation, kl, ks, kl_ks_variation)
        self.angular_velocity_factor = angular_velocity_factor

    def build_bounds(self):
        self.lower_bounds = MujocoArcMotionParameters()
        self.upper_bounds = MujocoArcMotionParameters()

    def update(self, xh_start, yh_start):
        DM = 1 / 10
        SMALL_EPS = 1e-5
        super().update(xh_start, yh_start)
        ball_radius = MujocoTrajectoryModel.BALL_RADIUS
        scale = self.angular_velocity_factor / (ball_radius / DM)
        self.lower_bounds.wx0 = self.lower_bounds.vx0 * scale - SMALL_EPS
        self.upper_bounds.wx0 = self.upper_bounds.vx0 * scale + SMALL_EPS
        self.lower_bounds.wy0 = self.lower_bounds.vy0 * scale - SMALL_EPS
        self.upper_bounds.wy0 = self.upper_bounds.vy0 * scale + SMALL_EPS
        self.lower_bounds.wz0 = self.lower_bounds.vz0 * scale - SMALL_EPS
        self.upper_bounds.wz0 = self.upper_bounds.vz0 * scale + SMALL_EPS
        self.lower_bounds.blunt_drag = self.lower_bounds.kl + self.DEFAULT_BLUNT
        self.upper_bounds.blunt_drag = self.upper_bounds.kl + self.DEFAULT_BLUNT
        self.lower_bounds.slender_drag = self.lower_bounds.ks + self.DEFAULT_SLENDER
        self.upper_bounds.slender_drag = self.upper_bounds.ks + self.DEFAULT_SLENDER
        self.lower_bounds.angular_drag = self.lower_bounds.k3 + self.DEFAULT_ANGULAR
        self.upper_bounds.angular_drag = self.upper_bounds.k3 + self.DEFAULT_ANGULAR
        self.lower_bounds.kutta_lift = self.lower_bounds.kl + self.DEFAULT_KUTTA
        self.upper_bounds.kutta_lift = self.upper_bounds.kl + self.DEFAULT_KUTTA
        self.lower_bounds.magnus_lift = self.lower_bounds.ks + self.DEFAULT_MAGNUS
        self.upper_bounds.magnus_lift = self.upper_bounds.ks + self.DEFAULT_MAGNUS

class MujocoTrajectoryModel(TrajectoryModel):
    BALL_MASS = 0.43
    BALL_RADIUS = 0.11  # m
    TIMESTEP = 0.005  # Simulation timestep for mujoco
    DM = 1/10

    motion_parameters_type = MujocoArcMotionParameters
    def __init__(self, use_ellipsoid_fluid=False, air_density=1.2,
                 air_viscosity=0.00002,
                 use_virtual_mass=True):
        self.air_density = air_density
        self.air_viscosity = air_viscosity
        self.use_virtual_mass = use_virtual_mass
        self.xml = create_xml(
            MujocoTrajectoryModel.BALL_MASS,
            MujocoTrajectoryModel.BALL_RADIUS,
            MujocoTrajectoryModel.TIMESTEP,
            self.air_density,
            self.air_viscosity
        )
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.timestep = self.get_timestep()
        self.use_ellipsoid_fluid = use_ellipsoid_fluid

    def simulate(self, initial_trajectory: Trajectory3D, motion_parameters: MujocoArcMotionParameters) -> Trajectory3D:
        """Gives the ball trajectory when moving in the air

        Parameters
        ----------
        r0: np.array
            shape (3, ), starting X,Y,Z coordinates of ball starting point
        v0: np.array
            shape (3,), starting velocity X,Y,Z coordinates
        w0: np.array
            shape (3,), starting angular velocity X,Y,Z coordinates
        t: np.array
            shape (T,) with T is number of frames, video frames per second
        g: float
            gravitational constant (normally 9.8)

        Returns
        -------
        rt: np.array
            shape (T, 3), trajectory coordinate vector for each time step
        """
        # initialize mujoco stuff
        data = mujoco.MjData(self.model)
        data.model.opt.gravity[2] = -motion_parameters.g

        # set initial conditions
        data.qpos[0] = motion_parameters.x0 * self.DM
        data.qpos[1] = motion_parameters.y0 * self.DM
        data.qpos[2] = motion_parameters.z0 * self.DM
        data.qvel[0] = motion_parameters.vx0 * self.DM
        data.qvel[1] = motion_parameters.vy0 * self.DM
        data.qvel[2] = motion_parameters.vz0 * self.DM
        data.qvel[3] = motion_parameters.wx0
        data.qvel[4] = motion_parameters.wy0
        data.qvel[5] = motion_parameters.wz0
        if self.use_ellipsoid_fluid:
            data.model.geom_fluid = self.get_geom_fluid(motion_parameters,
                                                        self.use_virtual_mass)
        else:
            data.model.geom_fluid = np.zeros(12)

        # simulate the movement of the ball
        rt = []
        times = []
        save_index = 0  # to remember the index of the next time (from initial_trajectory.time) to save the solution
        current_step = 0
        while save_index < len(initial_trajectory.time):
            # calculate number of steps until next save time
            # number of simulation steps needed to reach the next save time
            steps = round(
                (initial_trajectory.time[save_index] - data.time)
                / MujocoTrajectoryModel.TIMESTEP
            )
            mujoco.mj_step(self.model, data, steps)
            rt.append(data.qpos.copy())
            save_index += 1
            current_step += steps
            times.append(data.time)

        rt = np.array(rt)
        mujoco.mj_resetData(self.model, data)
        trajectory = Trajectory3D.from_xyz(rt[:, 0] / self.DM, rt[:, 1] / self.DM, rt[:, 2] / self.DM, initial_trajectory.time)
        trajectory.angle = rt[:, 3:]

        return trajectory
    
    def get_timestep(self):
        # get timestep from xml
        root = ET.fromstring(self.xml)
        timestep = None
        for child in root:
            if child.tag == "option":
                timestep = float(child.attrib["timestep"])
        assert timestep is not None, "Timestep not found in the xml"
        return timestep

    def get_geom_fluid(self, motion_parameters: MujocoArcMotionParameters,
                       use_virtual_mass=True):
        if use_virtual_mass:
            virtual_mass = 2/3 * np.pi *\
                           MujocoTrajectoryModel.BALL_RADIUS**3 * np.ones(3)
        else:
            virtual_mass = np.zeros(3)

        virtual_inertia = np.zeros(3)
        fluid_coefficients = np.array([
            1.,
            motion_parameters.blunt_drag,
            motion_parameters.slender_drag,
            motion_parameters.angular_drag,
            motion_parameters.kutta_lift,
            motion_parameters.magnus_lift,
        ])
        geom_fluid = np.concatenate([fluid_coefficients, virtual_mass, virtual_inertia])
        return geom_fluid


def create_xml(m, size, timestep, density=1.2, viscosity=0.00002):
    """Create the xml string for the mujoco simulation
    Note that the initial ball position is a random value and should be set later on!

    Parameters
    ----------
    m: float
        mass of the ball in kg
    size: float
        radius of the ball in meters
    timestep: float
        simulation timestep for mujoco

    Returns
    -------
    xml: str
        xml string for the mujoco simulation
    """
    xml = f"""
    <mujoco>
      <worldbody>
        <!-- Define the ball -->
        <body pos="5.425 0 1" name="ball">
            <joint type="free" name="ball_joint"/>
            <geom type="sphere" name="ball_geom" size="{size}" mass="{m}" pos="0 0 0" />
        </body>
      </worldbody>
      <!-- Enable gravity for the simulation -->
      <option gravity="0 0 -9.81" integrator="implicit" timestep="{timestep}" density="{density}" viscosity="{viscosity}" />
    </mujoco>
    """
    return xml

