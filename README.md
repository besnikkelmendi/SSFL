# SSFL
Small Scale Federated Learning


## Installation

Make sure Python V3+ is installed in your build environment to begin with and then install the libraries below:

- NumPy
- TensorFlow

## Configuration

- Variable **C** determines the fraction of clients that will be selected for training
- Variable **MIN_NO_CLIENTS** determines the minimum number of clients availbable for a training session to begin 

## Execution

1. Open any command line (terminal) window and run **python Server.py** to start the server
2. Open another terminal and run **python Client_Intance.py** to start a client 

Step two should be repeated until you instantiate at least **MIN_NO_CLIENTS** instances. 
