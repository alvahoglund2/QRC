# QRC - Quantum Dot Reservoir Computing 

QRC is a Python package designed for reservoir computing using quantum dot systems. This package provides tools for creating and simulating quantum reservoirs, training readout layers, and performing various computational tasks. 

Quantum Reservoir Computing Model
-------------------------------------------------------------------
The quantum reservoir computing model is consisting of two parts; a quantum reservoir and a readout layer. 

* **Qreservoir.py** The quantum reservoir is consisting of four coupled quantum dots connected to electronic leads. The system is simulated using QMEQ, an open source python package.

* **Qreadout.py**  The readout layer is a single-layer neural network, trained on the reservoir's output for some computational task. The readout layer is created and trained using Reservoirpy. 

Modules
--------------------------------------------------------------------
The QRC package consists of three different modules: 

* **Task_engine** Creates a quantum reservoir computing model that can be trained for various target functions.
* **Timing_task** Used for testing the quantum reservoir computing model for the timing task
* **Reservoir_dynamics**  Illustrates the dynamics within the quantum reservoir.

Tutorials
---------------------------------------------------------------------
Tutorials and examples of benchmark tests can be found in the folder "Tutorials". 

Installation
----------------------------------------------------------------------
To install the QRC package, clone the repository and install the required dependencies:

git clone https://github.com/alvahoglund2/QRC.git

cd QRC

pip install -r requirements.txt