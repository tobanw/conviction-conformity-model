# Network Simulation

## Requirements

- Python 2.7
- `pip`

## Installation

This assumes you're using Anaconda (and hence already have `networkx`, `numpy`, `scipy`, `pandas`, `matplotlib`).
`ComplexNetworkSim` and `SimPy` are in pypi, so we can install them with `pip`.
```
$ pip install ComplexNetworkSim
```
`ComplexNetworkSim` requires `SimPy` version 2:
```
$ pip install "simpy>=2.3,<3"
```

## Usage
Set parameters and output directory, then run
```
$ python sim-main.py
```
