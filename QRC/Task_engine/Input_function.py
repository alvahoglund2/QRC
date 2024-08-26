import numpy as np
from scipy.interpolate import interp1d
from reservoirpy.datasets import mackey_glass
from typing import Callable, Optional, Dict, Any


def set_input_func(input_func_name: str, **kwargs: Any) -> Callable:
    """
    Finds the input function for the specified input function name.
    Optional to add additional keyword arguments for the input function.
    """
    input_funcs = {
        "Random_sin": random_sin,
        "Random_box": random_box,
        "Random_smooth": random_smooth,
        "Random_white": random_white,
        "Sin": sin,
        "Sin_3": sin3,
        "Mackey_glass": mackey_glass_func,
    }
    if input_func_name in input_funcs:
        return input_funcs[input_func_name](**kwargs)
    else:
        raise ValueError("Input function not recognized")


def random_sin(
    seed: Optional[int] = None,
    t_def: np.ndarray = np.linspace(-1, 2, 3000),
    numb_terms: int = 1000,
    freq_min: float = 0.1,
    freq_max: float = 300,
    max_val: float = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generates a random function by summing sine functions with random amplitudes, frequencies and phases
    """
    # If seed is specified, set the seed for reproducibility
    if seed != None:
        np.random.seed(seed)

    result = np.zeros_like(t_def)

    # Ranndomize sine functions and sum them
    for i in range(numb_terms):
        amplitude = np.random.uniform(-1, 1)
        frequency = np.random.uniform(freq_min, freq_max)
        phase = np.random.uniform(0, 2 * np.pi)
        result += amplitude * np.sin(frequency * t_def + phase)

    # Normalize to [0,max_val]
    result_norm = ((result - result.min()) / (result.max() - result.min())) * max_val

    # Interpolate the function
    result_cont = interp1d(t_def, result_norm, kind="cubic")

    return result_cont


def sin3(
    seed: Optional[int] = None,
    t_def: np.ndarray = np.linspace(-1, 2, 3000),
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generates a function by summing sine functions with different amplitudes, frequencies and phases
    """
    result = np.zeros_like(t_def)

    # Parameters used for the sine function
    a = 2.11
    b = 3.73
    c = 4.11
    T = 1

    # Create the function
    result = (
        np.sin(2 * np.pi * a * t_def / T)
        * np.sin(2 * np.pi * b * t_def / T)
        * np.sin(2 * np.pi * c * t_def / T)
        + 1
    )
    # Normalize to [0,1]
    result_norm = (result - result.min()) / (result.max() - result.min())

    # Interpolate the function
    result_cont = interp1d(t_def, result_norm, kind="cubic")
    return result_cont


def random_box(
    seed: Optional[int] = None,
    t_def: np.ndarray = np.linspace(-1, 2, 18000),
    max_val: float = 1,
    box_points: int = 5,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generates a random function by randomizing values and smoothing with a box average
    """
    # If seed is specified, set the seed for reproducibility
    if seed != None:
        np.random.seed(seed)

    # Randomize values for the function
    steps = len(t_def)
    func = np.zeros(steps)

    for i in range(steps):
        func[i] = np.random.uniform(0, max_val)

    # Smooth the function with a box average
    box = np.ones(box_points) / box_points
    func_smooth = np.convolve(func, box, mode="same")
    func_cont = interp1d(t_def, func_smooth, kind="cubic")

    return func_cont


def random_smooth(
    seed: Optional[int] = None,
    t_def: np.ndarray = np.linspace(-1, 2, 500),
    max_val: float = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generates a random function by randomizing values and smoothing with a box average
    """
    # If seed is specified, set the seed for reproducibility
    if seed != None:
        np.random.seed(seed)

    # Randomize values for the function
    steps = len(t_def)
    func = np.zeros(steps)
    for i in range(steps):
        func[i] = np.random.uniform(0, max_val)

    # Interpolate the function
    func_cont = interp1d(t_def, func, kind="cubic")
    return func_cont


def random_white(
    seed: Optional[int] = None,
    t_def: np.ndarray = np.linspace(-1, 2, 3000),
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generates a random function of white noise
    """

    # If seed is specified, set the seed for reproducibility
    if seed != None:
        np.random.seed(seed)

    # Generate white noise
    white_noise = np.random.normal(0, 1, len(t_def))

    # Normalize to [0,1]
    white_noise = (white_noise - white_noise.min()) / (
        white_noise.max() - white_noise.min()
    )
    # Interpolate the function
    result_cont = interp1d(t_def, white_noise, kind="cubic")
    return result_cont


def sin(
    seed: Optional[int] = None, period_scale: float = 0.1
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a sine function with a specified period scaling.
    """

    def sin_function(t):
        return np.sin(2 * np.pi * t / period_scale)

    return sin_function


def mackey_glass_func(
    timesteps: int = 2510, tau: int = 17, seed: Optional[int] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Generates a Mackey-Glass time series
    """
    # Total number of steps extracted from the dataset
    timesteps_extended = 2510 * 3
    t_def = np.linspace(-1, 2, timesteps_extended)

    # Import the Mackey-Glass dataset
    mg = mackey_glass(timesteps_extended, tau).flatten()

    # Rescale to [0,1]
    mg = (mg - mg.min()) / (mg.max() - mg.min())

    # Interpolate the function
    func_cont = interp1d(t_def, mg, kind="cubic")
    return func_cont
