import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Tuple, Optional

sys.path.append("..//")
import Qreservoir


def get_charge_prob(reservoir: Qreservoir) -> np.ndarray:
    """
    Returns the probability for 0, 1, 2, 3 and 4 charges in the system

    Fetches  the time dependent density matrix the reservoir.
    By summing diagonal components of the density matrix, the charge probability over time is calculated.
    """
    if reservoir.phi_t is None:
        reservoir.get_phi_t()

    charge_prob_t = np.zeros((5, reservoir.phi_t.shape[1]))

    # Define the ranges for summing phi_t. Reservoir.split_points is used to split phi_t into components corresponding to different charges
    split_points = reservoir.split_points

    # Summing the diagonal components of the density matrix gives the charge probability
    for i in range(len(split_points) - 1):
        charge_prob_t[i, :] = np.sum(
            reservoir.phi_t[split_points[i] : split_points[i + 1], :], axis=0
        )
    return charge_prob_t


def get_state_b(reservoir: Qreservoir, b: int) -> List[Tuple[int, complex]]:
    """
    Returns the components of the many body state b in the single body basis.
    """
    srt = [0]
    qd = reservoir.system.qd
    si = reservoir.system.si
    charge, alpha = si.ind_qn[b]

    ind = si.chargelst[charge].index(b)
    coeffs = np.array(
        [abs(qd.vecslst[charge][:, ind]), range(len(si.chargelst[charge]))]
    )

    ind2 = np.lexsort([coeffs[i] for i in reversed(srt)])
    mult_arr = (coeffs.T[ind2]).T
    coeffs = np.array(mult_arr[1], dtype=int)
    fstates = []
    for j1 in reversed(coeffs):

        sn = si.chargelst[charge][j1]  # State number
        val = qd.vecslst[charge][j1, ind]  # Value
        fstates.append((sn, val))
    return fstates


def get_charge_dist_for_state(reservoir: Qreservoir, b: int) -> np.ndarray:
    """
    Input is the reservoir and the many body state b
    Returns a list with the probability of charge in each dot [QD1, QD2, QD3, QD4] for the specified state b
    """
    si = reservoir.system.si
    fstates = get_state_b(reservoir, b)
    charge_prob = np.zeros(reservoir.nbr_dots)
    for state in fstates:
        state_list = si.get_state(state[0])
        val = state[1]
        for i in range(len(state_list)):
            val_p = np.real(np.square(val))
            charge_prob[i] += state_list[i] * val_p
    return charge_prob


def get_charge_dist_for_all_states(reservoir: Qreservoir) -> np.ndarray:
    """
    Returns a list with the probability of charge in each dot [QD1, QD2, QD3, QD4] for all many body states
    """
    nbr_dots = reservoir.nbr_dots
    nbr_states = 2**nbr_dots
    charge_prob = np.zeros((nbr_states, nbr_dots))
    for i in range(nbr_states):
        charge_prob[i, :] = get_charge_dist_for_state(reservoir, i)
    return charge_prob


def get_charge_dist_t(reservoir: Qreservoir) -> np.ndarray:
    """
    Returns a list with the expectation value for the charge in each QD over time [QD1, QD2, QD3, QD4]
    """
    if reservoir.phi_t is None:
        reservoir.get_phi_t()
    phi_t = reservoir.phi_t
    charge_dist_t = np.zeros((reservoir.nbr_dots, len(reservoir.t_range)))
    charge_dist = get_charge_dist_for_all_states(reservoir)

    for i in range(charge_dist_t.shape[1]):
        phi_0 = phi_t[: 2**reservoir.nbr_dots, i]
        phi_0 = phi_0[:, np.newaxis]

        m1 = charge_dist * phi_0
        m2 = np.sum(m1, axis=0)
        charge_dist_t[:, i] = m2
    return charge_dist_t


def get_frequencies(reservoir: Qreservoir) -> List[np.ndarray]:
    """
    Gets the energies for the states and calculates the frequencies (energy differences).
    """
    # Get energies for the states
    Ea = reservoir.system.Ea
    # Split the energies into components corresponding to different charges
    split_points = reservoir.split_points

    # Calculate the frequencies (energy differences) for the states with the same charge
    frequencies = []
    for i in range(len(split_points) - 1):
        Ea_c = Ea[split_points[i] : split_points[i + 1]]
        f_c = []
        for i in range(len(Ea_c)):
            for j in range(i + 1, len(Ea_c)):
                f_c.append(Ea_c[j] - Ea_c[i])
        frequencies.append(np.array(f_c))
    return frequencies


def plot_charge_prob(reservoir: Qreservoir) -> None:
    """
    Plots charge probabilities over time, charge expectation value over time and charge distribution over time.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Get charge probabilities over time
    charge_prob_t = get_charge_prob(reservoir)

    # Plot charge expectation value over time
    charges = np.arange(5)
    charges = charges[:, np.newaxis]
    charge_exp_value = np.sum(charge_prob_t * charges, axis=0)
    axs[0].plot(reservoir.t_range, charge_exp_value)
    axs[0].set_title("Charge expectation value over time")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Charge")

    # Plot charge probabilities over time
    axs[1].set_title("Charge probabilities over time")
    for i in range(5):
        axs[1].plot(reservoir.t_range, charge_prob_t[i, :], label=str(i))
    axs[1].plot(
        reservoir.t_range,
        reservoir.reservoir_input(reservoir.t_range),
        "--",
        label="Input",
    )

    axs[1].legend()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Probability")

    # Plot charge distribution over time
    charge_dist_t = get_charge_dist_t(reservoir)

    for i in range(reservoir.nbr_dots):
        axs[2].plot(reservoir.t_range, charge_dist_t[i, :], label=f"QD{i+1}")
    axs[2].set_ylabel("Probability of charge in QD")
    axs[2].set_xlabel("Time")
    axs[2].legend()
    axs[2].set_title("Charge distribution in the system")

    fig.tight_layout()


def plot_states_prob(reservoir: Qreservoir) -> None:
    """
    Plots the probability of many body states (diagonal components of the density matrix) over time.
    """

    # Calculate the density matrix for reservoir if not already done
    if reservoir.phi_t is None:
        reservoir.get_phi_t()

    # Plot Input
    fig, axs = plt.subplots(6, 1, figsize=(10, 8))
    split_points = reservoir.split_points
    axs[0].plot(reservoir.t_range, reservoir.reservoir_input(reservoir.t_range))
    axs[0].set_title("Input")
    axs[0].set_xlabel("Time")

    # Plot the probability of many body states over time
    for i in range(len(split_points) - 1):
        for j in range(split_points[i], split_points[i + 1]):
            axs[i + 1].plot(reservoir.t_range, reservoir.phi_t[j, :], label=i)
            axs[i + 1].set_xlabel("Time")
            axs[i + 1].set_ylabel("Probability")
        axs[i + 1].set_title(f"{i} charge states")

    plt.suptitle("Probability of many body states over time")
    plt.tight_layout()


def plot_off_diag_states_prob(reservoir: Qreservoir) -> None:
    """
    Plots the off diagonal components of the density matrix over time.
    """

    # Calculate the density matrix for reservoir if not already done
    if reservoir.phi_t is None:
        reservoir.get_phi_t()

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot Input
    axs[0].plot(reservoir.t_range, reservoir.reservoir_input(reservoir.t_range))
    axs[0].set_title("Input")
    axs[0].set_xlabel("Time")

    npauli = (
        reservoir.system.si.npauli
    )  # Number of diagonal components in density matrix
    ndm0 = (
        reservoir.system.si.ndm0
    )  # [npauli: ndm] and [ndm:] Are the real and imaginary parts of the off diagonal components of the density matrix, respectively.

    # Real parts of off diagonal components
    for j in range(npauli, ndm0):
        axs[1].plot(reservoir.t_range, reservoir.phi_t[j, :])
        axs[1].set_xlabel("Time")
        axs[1].set_title("Real parts of off diagonal components")

    # Imaginary parts of off diagonal components
    for j in range(ndm0, reservoir.phi_t.shape[0]):
        axs[2].plot(reservoir.t_range, reservoir.phi_t[j, :])
        axs[2].set_xlabel("Time")
        axs[2].set_title("Imaginary parts of off diagonal components")

    fig.suptitle("Off diagonal components of the density matrix")
    fig.tight_layout()


def plot_frequencies(reservoir: Qreservoir) -> None:
    # Get frequencies
    frequencies = get_frequencies(reservoir)

    plt.figure(figsize=(8, 4))

    # Histogram parameters
    bins = 20
    alpha = 0.7

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    fig.suptitle("Frequencies for the closed system")

    # Plot all frequencies in histogram
    all_frequencies = np.concatenate(frequencies)
    axs[0].hist(
        np.concatenate(frequencies), bins=bins, alpha=alpha, label="All frequencies"
    )
    axs[0].set_title("All frequencies")
    axs[0].set_xlabel("Frequency")
    axs[0].set_xlim([0, np.max(all_frequencies)])

    # Plot frequencies for states with charge 0, 1, 2 and 3
    for i in range(1, 4):
        axs[i].hist(frequencies[i], bins=bins, alpha=alpha, label=" ")
        axs[i].set_title(f"Frequencies for states with charge {i}")
        axs[i].set_xlim([0, np.max(all_frequencies)])
        axs[i].set_xlabel("Frequency")

    plt.tight_layout()


def plot_energies(reservoir: Qreservoir) -> None:
    # Get energies for all states
    Ea = reservoir.system.Ea

    fig, axs = plt.subplots(6, 1, figsize=(10, 8))

    # Plot all energies in histogram
    axs[0].hist(Ea, bins=20, alpha=0.7)
    axs[0].set_title("Total energy distribution")
    axs[0].set_xlabel("Energy")
    axs[0].set_xlim([np.min(Ea) - 0.2, np.max(Ea) + 0.2])

    # Used to split the energies into components corresponding to different charges
    split_points = reservoir.split_points

    # Plot energies for states with charge 0, 1, 2 and 3
    for i in range(len(split_points) - 1):
        Ea_c = Ea[split_points[i] : split_points[i + 1]]
        axs[i + 1].set_xlim([np.min(Ea) - 0.2, np.max(Ea) + 0.2])
        axs[i + 1].hist(Ea_c, bins=20, alpha=0.7)
        axs[i + 1].set_title(f"Energy distribution for state with charge {i}")
        axs[i + 1].set_xlabel("Energy")
    fig.tight_layout()
