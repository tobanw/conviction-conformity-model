"""
Simulation: takes params and model specification
    - network structure
    - nodes are Agent objects
"""
# NOTE: this uses sequential dynamics -- agents act in turn, not simultaneously

# imports
from __future__ import division
import numpy as np
import matplotlib
import networkx as nx
from ComplexNetworkSim import NetworkSimulation, utils, PlotCreator
from ComplexNetworkSim import AnimationCreator
from agentlogic import MyAgent, Synchronizer

""" Hyperparameters
    - network size
    - number of actions
    - the network
"""
NODES = 150
num_acts = 5

# define public expressions, used as agent states
action_set = [ (2.0 * k + 1 - num_acts)/num_acts  for k in range(num_acts) ]

# Network topology
#G = nx.scale_free_graph(NODES)
#G = nx.erdos_renyi_graph(NODES,0.02)
G = nx.powerlaw_cluster_graph(NODES,3,0.8)


""" Parameter Combinations
    theta: can normalize so that it lies in the unit simplex
    gamma
"""
ngrid = 12 # simplex grid edge size: n/2 * (n-1) theta triplets (do 12)
# uniform grid on the interior of the unit simplex
theta_set = [ (x/ngrid, y/ngrid, 1-(x+y)/ngrid) for x in range(1,ngrid) for y in range(1,ngrid-x)]

gamma_set = range(2,5) # do 2-4

param_combos = [ { 'theta':theta, 'gamma':gamma, 'acts':action_set } 
                    for gamma in gamma_set for theta in theta_set ]


""" Simulation routine """

# Simulation constants
MAX_SIMULATION_TIME = 15.0
TRIALS = 1

sim_name = 'wide'
# initial states of agents: irrelevant since they'll start by playing belief
states = [0.0 for node in G.nodes()]

def main():

    # run simulation with parameters
    # - complex network structure
    # - initial state list
    # - agent behaviour class
    # - output directory
    # - maximum simulation time
    # - number of trials
    # - global shared parameters (kwd arg)
    for global_params in param_combos:
        directory = 'results/test/{gamma}--{theta[0]:.2}-{theta[1]:.2}-{theta[2]:.2}--{basename}'.format(
                theta = global_params['theta'],
                gamma = global_params['gamma'],
                basename = sim_name)
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
        myName = sim_name #name that you wish to give your image output files
        title = "Shares of positions expressed"
        statesToMonitor = action_set #even if we have states 0,1,2,3,... plot only 1 and 0
        colours = color_hex(action_set,color_mapper)
        #colours = ["blue", "gray", "red"] #state 1 in red, state 0 in green
        labels = [ "%1.2f" % act for act in action_set ]
        #labels = ["Left", "Neutral", "Right"] #state 1 named 'Infected', 0 named 'Susceptible'
        p = PlotCreator(directory, myName, title, statesToMonitor, colours, labels)
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


"""
Color map for plot
"""

# function that maps [0,1] to rgb proportions on blue-red spectrum
#color_mapper = matplotlib.colors.LinearSegmentedColormap.from_list('blue-red', ['blue','red'])

# define mapping from [0,1] to rgb proportions
gray_shade = 0.9
segmap = {'red':   [(0.0,  0.0, 0.0),
                   (0.5,  gray_shade, gray_shade),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (0.5,  gray_shade, gray_shade),
                   (1.0,  0.0, 0.0)],

         'blue':  [(0.0,  1.0, 1.0),
                   (0.5,  gray_shade, gray_shade),
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


# run main
if __name__ == '__main__':
    main()
