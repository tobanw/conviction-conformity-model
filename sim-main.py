"""
Model class: takes params and model specification
    - network structure
    - nodes are Agent objects
"""
# NOTE: this uses sequential dynamics -- agents act in turn, not simultaneously

# imports
import numpy as np
import matplotlib
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
num_acts = 5
action_set = [ (2.0 * k + 1 - num_acts)/num_acts  for k in range(num_acts) ]

# function that maps [0,1] to rgb proportions on blue-red spectrum
#color_mapper = matplotlib.colors.LinearSegmentedColormap.from_list('blue-red', ['blue','red'])

# define mapping from [0,1] to rgb proportions
gray_darkness = 0.9
segmap = {'red':   [(0.0,  0.0, 0.0),
                   (0.5,  gray_darkness, gray_darkness),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (0.5,  gray_darkness, gray_darkness),
                   (1.0,  0.0, 0.0)],

         'blue':  [(0.0,  1.0, 1.0),
                   (0.5,  gray_darkness, gray_darkness),
                   (1.0,  0.0, 0.0)]}
# function that maps [0,1] to rgb proportions on blue-gray-red spectrum
color_mapper = matplotlib.colors.LinearSegmentedColormap('blue-gray-red', segmap)

def color_hex(actions, colormap):
    raw_tuples = [ color_mapper( (act + 1)/2 ) for act in actions ]
    hex_values = []
    for rawcolor in raw_tuples:
        formatter = tuple([ 255 * rgbval for rgbval in rawcolor[:-1] ]) # strip alpha value
        hex_values.append('#%02x%02x%02x' % formatter)
    return hex_values


# initial states of agents: irrelevant since they'll start by playing belief
states = [0.0 for node in G.nodes()]
#states[0:25] = [LEFT]*25
#states[-25:] = [RIGHT]*25

# global params
THETA = (-3.3, -0.9, -1.6) # conviction, conformity, revision cost
GAMMA = (2,4)



### Simulation routine

# Simulation constants
MAX_SIMULATION_TIME = 25.0
TRIALS = 1
directory = STORAGE #output directory
global_params = {} # global dict of params
global_params['theta'] = THETA
global_params['gamma'] = GAMMA
global_params['acts'] = action_set

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
                                   **global_params)
    simulation.runSimulation()


    # plotting
    myName = "MORALS" #name that you wish to give your image output files
    title = "Simulation of moral expression network"
    statesToMonitor = action_set #even if we have states 0,1,2,3,... plot only 1 and 0
    colours = color_hex(action_set,color_mapper)
    #colours = ["blue", "gray", "red"] #state 1 in red, state 0 in green
    labels = [ "%1.2f" % act for act in action_set ]
    #labels = ["Left", "Neutral", "Right"] #state 1 named 'Infected', 0 named 'Susceptible'
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
