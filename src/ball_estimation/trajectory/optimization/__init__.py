from abc import abstractmethod, ABC
from copy import deepcopy
from ball_estimation.trajectory.model import ArcMotionParameters, MotionParameters, StraightMotionParameters
from ball_estimation.trajectory.loss_function import LossFunction, StraightLossFunction
from lmfit import Minimizer, Parameters
import numpy as np

class OptimizationBounds(ABC):
    def __init__(self, lower_bounds: MotionParameters, upper_bounds: MotionParameters):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

class OptimizationOptions(ABC):
    pass

class TrajectoryOptimizer(ABC):
    def __init__(self, optimization_options: OptimizationOptions, loss_function: LossFunction, 
                 optimization_bounds: OptimizationBounds | None = None):
        self.optimization_options = optimization_options
        self.loss_function = loss_function
        self.optimization_bounds = optimization_bounds

    @abstractmethod
    def optimize(self, motion_parameters_estimate: MotionParameters):
        pass


class ArcOptimizationBounds:
    def __init__(self, allowed_startpoint_shift_arc,
                 max_ball_speed, gravitation_g, g_variation,
                 k3, k3_variation, kl, ks, kl_ks_variation):
        self.allowed_startpoint_shift_arc = allowed_startpoint_shift_arc
        self.max_ball_speed = max_ball_speed
        self.gravitation_g = gravitation_g
        self.g_variation = g_variation
        self.k3 = k3
        self.k3_variation = k3_variation
        self.kl = kl
        self.ks = ks
        self.kl_ks_variation = kl_ks_variation

    def build_bounds(self):
        self.lower_bounds = ArcMotionParameters()
        self.upper_bounds = ArcMotionParameters()

    def update(self, xh_start, yh_start):
        self.xh_start = xh_start
        self.yh_start = yh_start

        min_bounds = self.lower_bounds
        max_bounds = self.upper_bounds
        min_bounds.x0 = self.xh_start - self.allowed_startpoint_shift_arc
        max_bounds.x0 = self.xh_start + self.allowed_startpoint_shift_arc
        min_bounds.y0 = self.yh_start - self.allowed_startpoint_shift_arc
        max_bounds.y0 = self.yh_start + self.allowed_startpoint_shift_arc
        min_bounds.z0 = 0.0
        max_bounds.z0 = 20.0
        min_bounds.vx0 = -self.max_ball_speed
        max_bounds.vx0 = self.max_ball_speed
        min_bounds.vy0 = -self.max_ball_speed
        max_bounds.vy0 = self.max_ball_speed
        min_bounds.vz0 = -0.5 * self.max_ball_speed
        max_bounds.vz0 = 0.5 * self.max_ball_speed
        min_bounds.k3 = max(0., self.k3 - self.k3_variation)
        max_bounds.k3 = self.k3 + self.k3_variation + 1e-6
        min_bounds.g = self.gravitation_g - self.g_variation
        max_bounds.g = self.gravitation_g + self.g_variation + 1e-6
        min_bounds.kl = -self.kl_ks_variation
        max_bounds.kl = self.kl_ks_variation + 1e-6
        min_bounds.ks = -self.kl_ks_variation
        max_bounds.ks = self.kl_ks_variation + 1e-6

        self.lower_bounds = min_bounds
        self.upper_bounds = max_bounds


class ArcOptimizer:
    def __init__(self, optimization_bounds: ArcOptimizationBounds,
                 optimization_options, loss_function):
        self.optimization_bounds = optimization_bounds
        self.optimization_options = optimization_options
        self.loss_function = loss_function

    def optimize(self, motion_parameters_estimate, extra_args):
        num_points, data, xh_end, yh_end, start_frame, endpoint = extra_args
        lower_bounds = self.optimization_bounds.lower_bounds.__dict__
        upper_bounds = self.optimization_bounds.upper_bounds.__dict__
        n_starts = max(int(getattr(self.optimization_options, "n_starts", 1)), 1)
        rng = np.random.default_rng()

        best_rss = np.inf
        best_dif = None
        best_values = None
        keys = motion_parameters_estimate.__dict__.keys()
        for i in range(n_starts):
            start_parameters = deepcopy(motion_parameters_estimate)
            if i > 0:
                for key in keys:
                    if key in self.optimization_options.vary.keys():
                        vary = self.optimization_options.vary[key]
                    else:
                        vary = True
                    if vary:
                        setattr(
                            start_parameters,
                            key,
                            rng.uniform(lower_bounds[key], upper_bounds[key]),
                        )

            params = Parameters()
            for key, value in start_parameters.__dict__.items():
                if key in self.optimization_options.vary.keys():
                    vary = self.optimization_options.vary[key]
                else:
                    vary = True
                params.add(
                    key,
                    value=value,
                    vary=vary,
                    min=lower_bounds[key],
                    max=upper_bounds[key],
                )

            minner = Minimizer(
                self.loss_function.fcn2min,
                params,
                fcn_args=(num_points, data, xh_end, yh_end, start_frame, endpoint),
            )
            kws = dict(self.optimization_options.minimize_kws)
            method = kws.pop("method", "least_squares")
            try:
                result = minner.minimize(method=method, **kws)
            except Exception as e:
                print(f"Exception occurred: {e}")
                return None, None, None
            dif = self.loss_function.fcn2min(
                result.params, num_points, data, xh_end, yh_end, start_frame, endpoint
            )
            rss = np.square(dif).sum()

            if best_values is None or rss < best_rss:
                best_rss = rss
                best_dif = dif
                best_values = {key: result.params[key].value for key in keys}

        params = Parameters()
        for key in keys:
            value = best_values[key]
            params.add(key, value)
            setattr(motion_parameters_estimate, key, value)
        err = np.sqrt((best_dif[:, 0] ** 2 + best_dif[:, 1] ** 2).sum() / len(best_dif))

        return (
            best_dif,
            err,
            motion_parameters_estimate
        )

class ArcOptimizationOptions:
    def __init__(self, vary, n_starts=1, minimize_kws: dict = None):
        self.vary = vary
        self.n_starts = n_starts
        self.minimize_kws = minimize_kws or {}

class StraightOptimizationBounds:
    def __init__(self, allowed_endpoints_shift_straight):
        self.allowed_endpoints_shift_straight = allowed_endpoints_shift_straight

        k_min = 0.0
        k_max = 0.15
        min_bounds = StraightMotionParameters(k_min, 1.0, 1.0, 1.0, 1.0)
        max_bounds = StraightMotionParameters(k_max, 1.0, 1.0, 1.0, 1.0)
        self.lower_bounds = min_bounds
        self.upper_bounds = max_bounds

    def update(self, x_start, y_start, x_end, y_end):
        self.lower_bounds.x_start = x_start - self.allowed_endpoints_shift_straight
        self.lower_bounds.y_start = y_start - self.allowed_endpoints_shift_straight
        self.lower_bounds.x_end = x_end - self.allowed_endpoints_shift_straight
        self.lower_bounds.y_end = y_end - self.allowed_endpoints_shift_straight
        self.upper_bounds.x_start = x_start + self.allowed_endpoints_shift_straight
        self.upper_bounds.y_start = y_start + self.allowed_endpoints_shift_straight
        self.upper_bounds.x_end = x_end + self.allowed_endpoints_shift_straight
        self.upper_bounds.y_end = y_end + self.allowed_endpoints_shift_straight

class StraightOptimizer:
    def __init__(self, loss_function: StraightLossFunction,
                 optimization_bounds: StraightOptimizationBounds):
        self.loss_function = loss_function
        self.optimization_bounds = optimization_bounds

    def optimize(self, motion_parameters: StraightMotionParameters, extra_args):
        num_points, data, start_frame = extra_args
        # find best parameters
        params = Parameters()

        # We allow a negative resistance coefficient here, which effectively corresponds
        # to the case of ball acceleration due to, for example, fast wind.
        lower_bounds = self.optimization_bounds.lower_bounds.__dict__
        upper_bounds = self.optimization_bounds.upper_bounds.__dict__
        for key, value in motion_parameters.__dict__.items():
            params.add(
                key,
                value=value,
                min=lower_bounds[key],
                max=upper_bounds[key],
            )
        minner = Minimizer(self.loss_function.fcn2min, params, fcn_args=(num_points, data, start_frame))
        try:
            result = minner.minimize(max_nfev=100)
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None, None, None, None, None, None, None

        # collect best parameters
        k_best = result.params["k"].value
        x_start_best = result.params["x_start"].value
        y_start_best = result.params["y_start"].value
        x_end_best = result.params["x_end"].value
        y_end_best = result.params["y_end"].value

        # apply best parameters
        params = Parameters()
        for key, value in {
            "k": k_best,
            "x_start": x_start_best,
            "y_start": y_start_best,
            "x_end": x_end_best,
            "y_end": y_end_best,
        }.items():
            params.add(key, value)
        dif = self.loss_function.fcn2min(params, num_points, data, start_frame)
        err = np.sqrt((dif[:, 0] ** 2 + dif[:, 1] ** 2).sum() / len(dif))
        return dif, err, k_best, x_start_best, y_start_best, x_end_best, y_end_best
