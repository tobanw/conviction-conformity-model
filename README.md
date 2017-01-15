# Network Simulation

## Requirements

- Anaconda
- Python 2.7
- `pip`

## Installation

This assumes you're using Anaconda (and hence already have `networkx`, `numpy`, `scipy`, `pandas`, `matplotlib`).
`ComplexNetworkSim` is in pypi, so you can install it with `pip`.

```
$ pip install ComplexNetworkSim
```

`ComplexNetworkSim` requires `SimPy` version 2, which can be installed from Anaconda:

```
$ conda install "simpy>=2.3,<3"
```

## Usage

Set parameters and output directory, then run

```
$ python sim-main.py
```

See the [`ComplexNetworkSim` docs](http://pythonhosted.org/ComplexNetworkSim/) to learn more.

## Animation example

`ComplexNetworkSim` provides an easy way to create an animated gif to vizualize a process on a network:

![random-network](https://cloud.githubusercontent.com/assets/667531/21959243/31cb0524-da8f-11e6-9603-05e46f1cc3e1.gif)
