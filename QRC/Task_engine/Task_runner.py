import numpy as np
import sys
from typing import Optional, Tuple, Any

sys.path.append("..\\")
from Qreservoir import Qreservoir
from Qreadout import Qreadout
import Input_function as input_func
import Target_function as target_func


class Task_runner:
    def __init__(
        self,
        input_func_name: str,
        evolution_rate: float,
        reservoir_seed: Optional[int] = None,
        input_func_seed: Optional[int] = None,
        V: int = 1,
        steps: int = 1000,
        input_weight_factor: float = 1.0,
        internal_weight_factor_o: float = 1.0,
        internal_weight_factor_c: float = 1.0,
        internal_weight_factor_e: float = 1.0,
        training_split: float = 2 / 3,
        regularization_term: float = 1e-6,
    ):
        """
        Initializes a Task_runner object. The task_runner object contains a reservoir and input function, giving a current.
        The current is calculated when the task_runner object is initialized.
        The Task_runner object can then be used to test the reservoir for different target functions.

        -  input_func_name specifies the input function type, all types can be found in class input_function.
        -  evolution_rate is the rate at which the system evolves, the rate rescales the kernel/Liouvillian. In the special case of 0, the system calculates the stationary current.
        -  reservoir_seed is the seed for the random generator of the reservoir parameters.
        -  input_func_seed is the seed for the random generator of the input function.
        -  V is the constant used for time multiplexing.
        -  steps is the number of time steps where the current is sampled.
        -  input_weight_factor is the scale of the input weights connecting the input function and chemical potentials.
        -  internal_weight_factor_o is the weight of the internal weights, corresponding to the interaction between quantum dots.
        -  internal_weight_factor_c is the weight of the internal weights, corresponding to the coulomb interaction between quantum dots.
        -  internal_weight_factor_e is the weight of the internal weights, corresponding to the energy levels of the quantum dots.
        -  training_split is the fraction of the data that is used for training the readout.
        -  regularization_term is the regularization term used in the readout layer
        """

        # The reservoir is evaluated in the time range [0, 1]
        self.t0 = 0  # Do not change this value
        self.tend = 1  # Must be positive

        self.steps = steps
        self.t_range = np.linspace(self.t0, self.tend, self.steps)[
            1:
        ]  # Remove the first timestep to make the time multiplexing work
        self.split_point = int(
            training_split * self.steps
        )  # defines where where data is split to train and test data
        self.t_train, self.t_test = self.split_data(self.t_range)
        self.input_weight_factor = input_weight_factor
        self.internal_weight_factor_o = internal_weight_factor_o
        self.internal_weight_factor_c = internal_weight_factor_c
        self.internal_weight_factor_e = internal_weight_factor_e
        self.evolution_rate = evolution_rate
        self.input_func = input_func.set_input_func(
            input_func_name, seed=input_func_seed
        )
        self.reservoir = None
        self.reservoir_seed = reservoir_seed
        self.regularization_term = regularization_term
        if evolution_rate == 0:
            self.I = self.get_current(stationary=True, V=V)
        else:
            self.I = self.get_current(stationary=False, V=V)

    def get_current(self, stationary: bool = False, V: int = 1) -> np.ndarray:
        """
        Creates the reservoir and generates the current for the input function.
        V is the constant used for time multiplexing/virtual nodes. For each time step, the current is sampled V times, giving V*nbr_dots readout weights.
        """

        # Increase the number of timesteps to implement time multiplexing
        steps_m = (self.steps - 1) * V + 1
        t_range_m = np.linspace(self.t0, self.tend, steps_m)

        # Create reservoir
        reservoir = Qreservoir(
            self.input_func,
            t_range_m,
            seed=self.reservoir_seed,
            evolution_rate=self.evolution_rate,
            input_weight_factor=self.input_weight_factor,
            internal_weight_factor_o=self.internal_weight_factor_o,
            internal_weight_factor_c=self.internal_weight_factor_c,
            internal_weight_factor_e=self.internal_weight_factor_e,
        )
        self.reservoir = reservoir

        # Get current from reservoir
        if stationary:
            I = reservoir.get_I_stat_t()
        else:
            I = reservoir.get_I_t()

        # Remove first timestep to make the time multiplexing work
        I = I[1:]

        # Reshape the current to the number of steps giving shape (steps-1, nbr_dots*V)
        I = I.reshape(self.steps - 1, V, reservoir.nbr_dots)
        I_extended = I.reshape(self.steps - 1, V * reservoir.nbr_dots)

        # Rescale the current to maximum absolute value 1
        max_val = np.max(I_extended)
        I_rescaled = I_extended / max_val
        return I_rescaled

    def run_test(
        self,
        target_func_name,
        target_param: Optional[Any] = None,
        warmup: int = 100,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Tests the reservoir with a target function.
        The target function is specified with a name and a parameter, all possible target functions can be found in Target_function.py.
        """

        # Get target function
        target = target_func.set_target(
            target_func_name, target_param, self.t_range, self.input_func
        )

        # Split data into train and test data
        train_target, test_target = self.split_data(target)
        train_I, test_I = self.split_data(self.I)

        # Create readout layer, train it on training data and get predictions on test data
        regularization_term = self.regularization_term
        readout = Qreadout(regularization_term)
        predictions = readout.get_predictions(train_I, train_target, test_I, warmup)

        # Calculate memory accuracy and normalized mean squared error by comparing target and predictions
        ma = self.memory_accuracy_measurement(predictions, test_target)
        nmse = self.nmse(predictions, test_target)

        return ma, nmse, predictions, test_target

    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the data into train and test data
        """
        training_data = data[: self.split_point]
        test_data = data[self.split_point :]
        return training_data, test_data

    def memory_accuracy_measurement(
        self, predictions: np.ndarray, y_target: np.ndarray
    ) -> float:
        """
        Calculates the memory accuracy of the prediction
        """
        cov_matrix = np.cov(predictions.flatten(), y_target)
        print("matrix:", cov_matrix)
        cov_pred_target = cov_matrix[0, 1]
        var_pred = cov_matrix[0, 0]
        var_target = cov_matrix[1, 1]
        ma = (cov_pred_target**2) / (var_pred * var_target)
        return ma

    def nmse(self, predictions: np.ndarray, y_target: np.ndarray) -> float:
        """
        Calculates the normalized mean squared error of the prediction
        """
        nmse = np.mean((predictions.flatten() - y_target) ** 2)
        variance = np.var(y_target)
        return nmse / variance if variance != 0 else 0
