"""
Model class: takes params and model specification
    - network structure
    - nodes are Agent objects
Agent class:
    - utility / cost function params
    - decision rule (focs)
    - attibutes: self-consciousness, 
    - state: private conviction
"""
# NOTE: this uses sequential dynamics -- agents act in turn, not simultaneously

# imports
import pandas as pd
import networkx as nx
from ComplexNetworkSim import NetworkSimulation, utils, PlotCreator
from ComplexNetworkSim import AnimationCreator
from agentlogic import MyAgent, Synchronizer

# output directory
STORAGE = 'results/test'

# Number of nodes 
NODES = 150

# Network topology
#G = nx.scale_free_graph(NODES)
G = nx.erdos_renyi_graph(NODES,0.02)

# define public expressions, used as agent states
LEFT = -2.0/3
NEUTRAL = 0.0
RIGHT = 2.0/3

# initial states of agents: irrelevant since they'll start by playing belief
states = [NEUTRAL for node in G.nodes()]
#states[0:25] = [LEFT]*25
#states[-25:] = [RIGHT]*25

# global params
THETA = (-3.3, -0.9, -1.6) # conviction, conformity, revision cost
GAMMA = (2,4)



### Simulation routine

# Simulation constants
MAX_SIMULATION_TIME = 40.0
TRIALS = 1
directory = STORAGE #output directory
globalSharedParameters = {} # global dict of params
globalSharedParameters['theta'] = THETA
globalSharedParameters['gamma'] = GAMMA
globalSharedParameters['acts'] = [LEFT,NEUTRAL,RIGHT]

def main():

    # run simulation with parameters
    # - complex network structure
    # - initial state list
    # - agent behaviour class
    # - output directory
    # - maximum simulation time
    # - number of trials
    # - global shared parameters (kwd arg)
    simulation = NetworkSimulation(G,
                                   states,
                                   MyAgent,
                                   directory,
                                   MAX_SIMULATION_TIME,
                                   TRIALS,
                                   Synchronizer,
                                   **globalSharedParameters)
    simulation.runSimulation()


    # plotting
    myName = "MORALS" #name that you wish to give your image output files
    title = "Simulation of moral expression network"
    statesToMonitor = [LEFT, NEUTRAL, RIGHT] #even if we have states 0,1,2,3,... plot only 1 and 0
    colours = ["blue", "gray", "red"] #state 1 in red, state 0 in green
    labels = ["Left", "Neutral", "Right"] #state 1 named 'Infected', 0 named 'Susceptible'
    p = PlotCreator(STORAGE, myName, title, statesToMonitor, colours, labels)
    # save png
    p.plotSimulation(show=False)

    """ don't plot network for now
    mapping = {LEFT:"blue", RIGHT:"red", NEUTRAL:"gray"}
    trialToVisualise = 0
    # create png
    visualiser = AnimationCreator(STORAGE, myName, title, mapping,
                              trial=trialToVisualise, delay=30)
    # create animated gif
    visualiser.create_gif(verbose=True)
    """


# run main
if __name__ == '__main__':
    main()
