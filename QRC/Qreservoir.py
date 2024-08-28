import qmeq
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from typing import Callable, List, Optional, Tuple


class Qreservoir:
    def __init__(
        self,
        reservoir_input: Callable[[float], float],
        t_range: List[float],
        seed: Optional[int] = None,
        evolution_rate: float = 1,
        input_weight_factor: float = 1,
        internal_weight_factor_o: float = 1,
        internal_weight_factor_c: float = 1,
        internal_weight_factor_e: float = 1,
    ) -> None:
        """
        Initializes a quantum reservoir object with 4 quantum dots (QD) and 4 leads.
        An input function is fed into the reservoir by changing the chemical potential.
        The output from the reservoir is the current from each lead.
        The current is calculated by solving the Lindblad equation with a RK23 ODE solver

        split_points: Used to split the density matrix (phi_t) into components corresponding to different charges
        internal_weight_factor_e: The weight determining in what range the QD energies are initialized in
        internal_weight_factor_c: The weight determining in what range the QD coulomb interaction are initialized in
        internal_weight_factor_o: The weight determining in what range the tunneling amplitudes between QDs are initialized in
        input_weight_factor: The scale of the input weights connecting the input function and chemical potentials

        reservoir_input: A time dependent function that is fed into the reservoir
        t_range: The time range for which the system is solved
        seed: Seed for random generator of the reservoir parameters
        evolution_rate: The rate at which the system evolves, the rate rescales the kernel/Liouvillian
        phi_t: The state (density matrix) of the system at each time step
        system: The qmeq system object
        kerns: The kernel/Liouvillian at each time step
        """
        self.solve_time = (
            0  # Used to keep track of how long it takes to solve the system
        )

        self.nbr_dots = 4
        self.split_points = [
            0,
            1,
            5,
            11,
            15,
            16,
        ]  # Used to split the phi_t array into the different charges
        self.input_weight_factor = input_weight_factor
        self.internal_weight_factor_o = internal_weight_factor_o
        self.internal_weight_factor_c = internal_weight_factor_c
        self.internal_weight_factor_e = internal_weight_factor_e

        self.reservoir_input = reservoir_input
        self.t_range = t_range
        self.evolution_rate = evolution_rate
        self.seed = seed

        self.phi_t = None
        self.w_input = None
        self.system = self.create_system()
        self.kerns = []
        self.I = None

    def create_system(self) -> qmeq.Builder:
        """
        Defines parameters and creates a QMEQ system object with 4 quantum dots and 4 leads.
        The system is modelled with the Hamiltonian:
        H = H_leads + H_tunneling + H_dot where H_dot = H_single + H_coulomb

        The following parameters are defined:

        nsingle - number of single particles states
        hsingle - singel particle Hamiltonian
        coulomb - Coulomb matrix elements
        nleads - number of the leads
        tleads - single particle tunneling amplitudes
        mulst - chemical potentials of the leads
        tlst - temperatures of the leads
        dband - bandwidht of the leads
        kernel - type of kernel to solve system

        More information about can be found in the QMEQ documentation
        """
        # Makes the random generator deterministic if seed is set
        if self.seed is not None:
            random.seed(self.seed)

        # --------- Input weight parameters ------------
        # Weights for the input function to the chemical potential in leads
        # The weights are currently hard coded but can be changed here
        w0_in = 1
        w1_in = -1
        w2_in = -2
        w3_in = 4

        self.w_in = np.array([w0_in, w1_in, w2_in, w3_in]) * self.input_weight_factor

        # --------- Lead and tunneling parameters, H_leads and H_tunneling ------------

        # The chemical potential at time 0 determined by the input function
        initial_value = self.get_mu_at_t(0)
        mu_lst0, mu_lst1, mu_lst2, mu_lst3 = initial_value

        gam = 1  # Gamma - tunneling rate between QDs and leads
        temp = 4.0  # Temperature
        dband = np.power(10.0, 4)  # Bandwidht of the leads

        tunneling_leads = np.sqrt(
            gam / (2 * np.pi)
        )  # Tunneling amplitude between QDs and leads
        tleads = {
            (0, 0): tunneling_leads,
            (1, 1): tunneling_leads,
            (2, 2): tunneling_leads,
            (3, 3): tunneling_leads,
        }

        nleads = self.nbr_dots  # Number of leads

        mulst = {
            0: mu_lst0,
            1: mu_lst1,
            2: mu_lst2,
            3: mu_lst3,
        }  # Chemical potentials of the leads
        tlst = {0: temp, 1: temp, 2: temp, 3: temp}  # Temperatures of the leads

        # --------- Quantum dot single-particle Hamiltonian, H_single ------------

        nsingle = self.nbr_dots  # Number of single particle states

        # The range of the QD energies
        e_min, e_max = (
            -1 * self.internal_weight_factor_e,
            1 * self.internal_weight_factor_e,
        )

        # The range of the QD coulomb interactions
        o_min, o_max = (
            0,
            1 * self.internal_weight_factor_o,
        )

        # Randomly initialize the QD energies and coulomb interactions

        e0, e1, e2, e3 = (
            random.uniform(e_min, e_max),
            random.uniform(e_min, e_max),
            random.uniform(e_min, e_max),
            random.uniform(e_min, e_max),
        )
        o01, o02, o03, o12, o13, o23 = (
            random.uniform(o_min, o_max),
            random.uniform(o_min, o_max),
            random.uniform(o_min, o_max),
            random.uniform(o_min, o_max),
            random.uniform(o_min, o_max),
            random.uniform(o_min, o_max),
        )

        # Singel Hamiltonian
        hsingle = np.array(
            [
                [e0, o01, o02, o03],
                [o01, e1, o12, o13],
                [o02, o12, e2, o23],
                [o03, o13, o23, e3],
            ]
        )

        # --------- Quantum dot Coulomb interaction, H_coulomb ------------

        # The range of the QD coulomb interactions
        c_min, c_max = (0, 1 * self.internal_weight_factor_c)

        # Randomly initialize the QD coulomb interactions
        coulomb = {
            (0, 1, 1, 0): random.uniform(c_min, c_max),
            (2, 3, 3, 2): random.uniform(c_min, c_max),
            (0, 2, 2, 0): random.uniform(c_min, c_max),
            (0, 3, 3, 0): random.uniform(c_min, c_max),
            (1, 2, 2, 1): random.uniform(c_min, c_max),
            (1, 3, 3, 1): random.uniform(c_min, c_max),
        }

        # --------- Create the QMEQ system object ------------
        system = qmeq.Builder(
            nsingle,
            hsingle,
            coulomb,
            nleads,
            tleads,
            mulst,
            tlst,
            dband,
            kerntype="Lindblad",
        )

        # Solve syestem
        system.make_kern_copy = True
        system.solve()

        return system

    def get_mu_at_t(self, t: float) -> Tuple[float, float, float, float]:
        """
        Calculates the chemical potential at time t based on the input function
        """
        w_in = self.w_in
        y = self.reservoir_input(t)
        mu0 = y * w_in[0]
        mu1 = y * w_in[1]
        mu2 = y * w_in[2]
        mu3 = y * w_in[3]
        return mu0, mu1, mu2, mu3

    def get_I_from_phi(self, phi: np.ndarray, t: float) -> np.ndarray:
        """
        Calculates the current at time t based on the state phi
        """
        self.system.phi0 = phi  # Get the current state of the system

        mu_lst0, mu_lst1, mu_lst2, mu_lst3 = self.get_mu_at_t(
            t
        )  # Get the chemical potential at time t
        self.system.change(
            mulst={0: mu_lst0, 1: mu_lst1, 2: mu_lst2, 3: mu_lst3}
        )  # Change the chemical potential of the leads

        # Generate the current for the given chemical potential and state
        self.system.appr.generate_fct()
        self.system.appr.kernel_handler.set_phi0(phi)
        self.system.current.fill(0.0)
        self.system.appr.generate_current()

        return np.array(self.system.current)

    def get_kern_at_t(self, t: float) -> np.ndarray:
        """
        Calculates the kernel at time t based on the chemical potential
        """
        mu_lst0, mu_lst1, mu_lst2, mu_lst3 = self.get_mu_at_t(
            t
        )  # Get the chemical potential at time t

        # Time the time it takes to solve the system
        solve_start_time = time.time()

        # Change the chemical potential of the leads
        self.system.change(mulst={0: mu_lst0, 1: mu_lst1, 2: mu_lst2, 3: mu_lst3})

        # Generate kernel
        self.system.appr.prepare_kern()
        self.system.appr.generate_fct()
        self.system.appr.generate_kern()
        kern = self.system.kern
        self.kerns.append(kern)

        solve_end_time = time.time()
        self.solve_time += solve_end_time - solve_start_time
        return kern

    def ode(self, t: float, phi: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of the state at time t and rescales it with the evolution rate
        The ode to solve is dphi/dt = kern * phi
        """
        kern = self.get_kern_at_t(t) * self.evolution_rate
        dphi_dt = np.dot(kern, phi)
        return dphi_dt

    def get_phi_t(self) -> np.ndarray:
        """
        Solves the system for the given time range, returns the state (density matrix) at each time step
        The initial state is the stationary state with the chemical potential at time 0
        """
        # Solve QMEQ system
        self.system.solve()
        phi0 = self.system.phi0.copy()

        # Solve ODE with RK23
        ode_start_time = time.time()
        solution = solve_ivp(
            self.ode,
            [self.t_range[0], self.t_range[-1]],
            phi0,
            t_eval=self.t_range,
            method="RK23",
            atol=1e-6,
            rtol=1e-6,
        )
        ode_end_time = time.time()

        # Save the state at each time step
        phi_t = solution.y
        self.phi_t = phi_t

        # Print information about the time it took to solve the system
        print("Time to solve ode: ", ode_end_time - ode_start_time)
        print("Number of iterations (nfev): ", solution.nfev)
        print("Total time to solve system: ", self.solve_time)
        print("Average time to solve system", self.solve_time / solution.nfev)
        return phi_t

    def get_I_t(self) -> np.ndarray:
        """ "
        Calculates the current at each time step based on the state at each time step
        """
        # Get the state at each time step
        phi_t = self.get_phi_t()

        # Calculate the current at each time step based on the state
        I_all = np.zeros((phi_t.shape[1], self.nbr_dots))
        for i in range(phi_t.shape[1]):
            I_all[i, :] = self.get_I_from_phi(phi_t[:, i], self.t_range[i])
        self.I = I_all
        return I_all

    def get_I_stat_t(self) -> np.ndarray:
        """
        Calculates the stationary current for the system at each time step using QMEQ function
        """
        I_list = []
        for t in self.t_range:
            mu_lst0, mu_lst1, mu_lst2, mu_lst3 = self.get_mu_at_t(
                t
            )  # Get the chemical potential at time t
            self.system.change(
                mulst={0: mu_lst0, 1: mu_lst1, 2: mu_lst2, 3: mu_lst3}
            )  # Change the chemical potential of the leads
            self.system.solve()  # Solve the system
            I = self.system.current.copy()  # Get the stationary current
            I_list.append(I)
        return np.array(I_list)
