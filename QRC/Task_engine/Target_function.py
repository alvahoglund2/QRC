import numpy as np
from typing import Callable, Dict, Any


def set_target(
    target_func_name: str,
    target_param: Any,
    t_range: np.ndarray,
    input_func: Callable[[float], float],
) -> np.ndarray:
    """
    Finds the target function for the specified target function name
    """
    target_funcs = {
        "Delay": lambda: time_delay(t_range, input_func, delay=target_param),
        "Narma": lambda: narma_n(t_range, input_func, n=target_param),
        "Prediction": lambda: prediction(
            t_range, input_func, prediction_steps=target_param
        ),
    }

    if target_func_name not in target_funcs:
        raise ValueError("Target function name not recognized")

    if target_param is None:
        raise ValueError(f"{target_func_name} target function requires a parameter")

    return target_funcs[target_func_name]()


def narma_n(
    t_range: np.ndarray,
    input_func: Callable[[float], float],
    n: int,
    alfa: float = 0.3,
    beta: float = 0.01,
    gamma: float = 1.5,
    delta: float = 0.1,
) -> np.ndarray:
    """
    Generates a "Non-linear autoregressive moving average" (NARMA) target function.
    """
    # Defing the time range for generating the target function. Need to add t_range_init to initialize the function
    steps_init = len(t_range) - 1
    t_range_init = -t_range[1 : steps_init + 1][::-1]
    t_range_tot = np.concatenate([t_range_init, t_range])

    # Sample points for the input function
    input = input_func(t_range_tot)

    # Rescale input to range [0, 0.2] to avoid divergence
    min_val = np.min(input)
    max_val = np.max(input)
    input = 0.2 * (input - min_val) / (max_val - min_val)

    # Generate the target function
    target_tot = np.zeros(len(t_range_tot))

    for t_idx in range(n, len(target_tot) - 1):
        sum = np.sum(target_tot[t_idx - n : t_idx])
        target_tot[t_idx + 1] = (
            alfa * target_tot[t_idx]
            + beta * target_tot[t_idx] * sum
            + gamma * input[t_idx - n] * input[t_idx - 1]
            + delta
        )

    # Remove the initial values
    target = target_tot[steps_init:]

    # Normalize the target
    target = ((target - target.min()) / (target.max() - target.min())) * 2 - 1
    return target


def time_delay(
    t_range: np.ndarray, input_func: Callable[[float], float], delay: int
) -> np.ndarray:
    """
    Generates a time delay target function, where the target is the input delayed by delay
    """

    # Defining the time range for generating the target function. Need to add t_range_init to initialize the function
    steps_init = len(t_range) - 1
    t_range_init = -t_range[1 : steps_init + 1][::-1]
    t_range_tot = np.concatenate([t_range_init, t_range])

    # Generate the input function
    input = input_func(t_range_tot)

    # Generate the target function. The target is the input delayed by delay
    target = np.roll(input, delay)

    # Remove the initial values
    target = target[steps_init:]

    return target


def prediction(
    t_range: np.ndarray,
    input_func: Callable[[float], float],
    prediction_steps: int,
) -> np.ndarray:
    """
    Generates target function that is prediction_steps ahead of the input function, where the target is the input shifted by prediction_steps
    """
    # Defining the time range for generating the target function. Need to add t_range_end to initialize the function
    step_size = t_range[1] - t_range[0]
    new_end_value = t_range[-1] + len(t_range) * step_size
    t_range_tot = np.arange(t_range[0], new_end_value + step_size, step_size)

    # Generate the input function
    input = input_func(t_range_tot)

    # Generate the target function. The target is the input shifted by prediction_steps
    target = np.roll(input, -prediction_steps)

    # Remove the end values
    target = target[: len(t_range)]

    return target
