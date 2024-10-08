import numpy as np
from scipy.interpolate import interp1d
import sys
from typing import Optional, Tuple, Callable

sys.path.append("..\\")
from Qreadout import Qreadout
from Qreservoir import Qreservoir


class Timing_task:
    def __init__(
        self,
        evolution_rate: float,
        steps: int = 1000,
        reservoir_seed: Optional[int] = None,
        input_weight_factor: float = 1.0,
        internal_weight_factor_o: float = 1.0,
        internal_weight_factor_c: float = 1.0,
        internal_weight_factor_e: float = 1.0,
    ):
        """
        In the timing task, a step function is used as input. The task is to output a peak a certain delay after the step.

        Initializes a Timing_task object. The object contains a reservoir and a current is generated for the input function (a step function).
        The Timing_task object can then be used to test the reservoir for the timing task with different timers.

        The same output from the reservoir is used for both training and testing.
        """
        # The reservoir is evaluated in the time range [0, 1]
        self.t0 = 0  # Do not change this value
        self.tend = 1  # Must be positive
        self.steps = steps  # Number of time steps where the current is sampled
        self.t_range = np.linspace(self.t0, self.tend, self.steps)  # Time range
        self.switch_location = steps // 3  # Location of the step function
        self.input_func = self.get_input_func()  # Input function
        self.evolution_rate = (
            evolution_rate  # Evolution rate, rescales the kernel/Liouvillian
        )
        self.reservoir_seed = (
            reservoir_seed  # Seed for the random generator of the reservoir parameters
        )
        self.input_weight_factor = input_weight_factor  # Scale of the input weights connecting the input function and chemical potentials
        self.internal_weight_factor_o = internal_weight_factor_o  # Weight of the internal weights, corresponding to the interaction between quantum dots
        self.internal_weight_factor_c = internal_weight_factor_c  # Weight of the internal weights, corresponding to the coulomb interaction between quantum dots
        self.internal_weight_factor_e = internal_weight_factor_e  # Weight of the internal weights, corresponding to the energy levels of the quantum dots
        self.reservoir = None  # Reservoir object
        self.I = self.get_current()  # Current generated by the reservoir

    def get_input_func(self) -> Callable[[float], float]:
        """
        Returns the step function used as input for the timing task
        """
        input_arr = np.zeros(self.steps)
        input_arr[self.switch_location :] = 1

        # Interpolate the step function
        func_input = interp1d(self.t_range, input_arr)

        return func_input

    def get_target(self, timer: int) -> np.ndarray:
        """
        Returns the target function for the timing task, where the target is a peak a certain delay (timer) after the step function
        """
        pulse_duration = 10  # The width of the peak
        target_arr = np.zeros(self.steps)
        target_arr[
            self.switch_location + timer : self.switch_location + timer + pulse_duration
        ] = 1
        return target_arr

    def get_current(self) -> np.ndarray:
        """
        Creates the reservoir and generates the current for the input function
        """
        reservoir = Qreservoir(
            self.input_func,
            self.t_range,
            evolution_rate=self.evolution_rate,
            seed=self.reservoir_seed,
            input_weight_factor=self.input_weight_factor,
            internal_weight_factor_c=self.internal_weight_factor_c,
            internal_weight_factor_e=self.internal_weight_factor_e,
            internal_weight_factor_o=self.internal_weight_factor_o,
        )
        self.reservoir = reservoir
        I = reservoir.get_I_t()

        return I

    def run_test(
        self, timer: int, warmup: Optional[int] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Tests the reservoir for the timer task with a specified timer value.
        """
        # Set warmup if not specified
        if warmup is None:
            warmup = self.steps // 10

        # Get the target function
        target = self.get_target(timer)

        # Normalize the current so that all values are between -1 and 1
        min_val = np.min(self.I)
        max_val = np.max(self.I)
        I_rescaled = 2 * (self.I - min_val) / (max_val - min_val) - 1

        # Train readout layer and get predictions
        readout = Qreadout()
        predictions = readout.get_predictions(I_rescaled, target, I_rescaled, warmup)

        # Measure performance
        ma = self.memory_accuracy_measurment(predictions, target)
        mse = self.mse(predictions, target)
        return ma, mse, predictions, target

    def memory_accuracy_measurment(
        self, predictions: np.ndarray, y_target: np.ndarray
    ) -> float:
        """
        Calculates the memory accuracy of the prediction:
        memory accuracy = (cov(pred, target)^2) / (var(pred) * var(target))
        """
        cov_matrix = np.cov(predictions.flatten(), y_target)
        cov_pred_target = cov_matrix[0, 1]
        var_pred = cov_matrix[0, 0]
        var_target = cov_matrix[1, 1]
        ma = (cov_pred_target**2) / (var_pred * var_target)
        return ma

    def mse(self, predictions: np.ndarray, y_target: np.ndarray) -> float:
        """
        Calculates the mean squared error of the prediction
        """
        return np.mean((predictions.flatten() - y_target) ** 2)
