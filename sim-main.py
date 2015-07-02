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
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import fsolve
from ComplexNetworkSim import NetworkAgent, Sim, NetworkSimulation, utils, PlotCreator, AnimationCreator

# output directory
STORAGE = sys.argv[1]

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


### Agent behavior

class MyAgent(NetworkAgent):
    """ an implementation of an agent """

    def __init__(self, state, initialiser):
        NetworkAgent.__init__(self, state, initialiser) # doesn't accept stateVector

        self.theta = self.globalSharedParameters['theta']
        self.gamma = self.globalSharedParameters['gamma']
        # TODO try truncated normal instead of uniform distro of initial views
        self.stateVector = 2 * self.r.random() - 1 # private belief, uniform in [-1,1]
        self.local_avg = 0.0 # initialise the attribute

        # start off agents acting according to their beliefs
        # NOTE this might not work if they are initialized and run one by one
        if self.stateVector < -1.0/3:
            self.state = LEFT
        elif self.stateVector > 1.0/3:
            self.state = RIGHT
        else:
            self.state = NEUTRAL

    def Run(self):
        while True:
            self.maximize()
            # wait 1 step before this agent gets run again
            yield Sim.hold, self, NetworkAgent.TIMESTEP_DEFAULT

    def maximize(self):
        """
        Solve for optimal action (which is self.state)
        """
        acts = [LEFT,NEUTRAL,RIGHT]

        # get local_avg action once to economize on computation
        self.set_local_avg()

        maxima = np.array([ self.theta[0] * (a - self.updated_view(self.optim_lam(a)))**2
                            + self.theta[1] * self.ms_dev(a)
                            + self.theta[2] * self.optim_lam(a)**self.gamma[0] *
                              self.polar_cost()  for a in acts ])
        globalMaxIndex = np.argmax(maxima)

        # set action
        self.state =acts[globalMaxIndex]

        # set belief
        self.stateVector = self.updated_view( self.optim_lam(acts[globalMaxIndex]) )

    def optim_lam(self, a):
        """
        solve non-linear foc for lambda, given discrete `a`
        """
        foc_lam = ( lambda lam: self.theta[2]*
                ( self.gamma[0] * lam**(self.gamma[0]-1) * self.polar_cost() 
                    + lam**self.gamma[0] * 2 * (self.local_avg -
                        self.stateVector ) )
                - 2 * self.theta[0] * (a - self.updated_view(lam) ) *
                (self.local_avg - self.stateVector) )
        return fsolve( foc_lam, 0.5)[0]

    def updated_view(self, lam):
        return (1-lam)*self.stateVector + lam * self.local_avg

    def set_local_avg(self):
        nbrs = self.getNeighbouringAgentsIter()
        # TODO edge weights (here is just using unweighted)
        self.local_avg = np.mean([nb.state for nb in nbrs])
        return

    def ms_dev(self, a):
        nbrs = self.getNeighbouringAgentsIter()
        return np.mean([(a - nb.state)**2 for nb in nbrs])

    def polar_cost(self):
        # changing views more costly when avg is far
        kernel = lambda mu: abs(self.local_avg - mu)**self.gamma[1]
        return kernel(self.stateVector)



### Simulation routine

# Simulation constants
MAX_SIMULATION_TIME = 10.0
TRIALS = 1

def main():
    directory = STORAGE #output directory
    globalSharedParameters = {} # arbitrary dict
    globalSharedParameters['theta'] = THETA
    globalSharedParameters['gamma'] = GAMMA

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

    mapping = {LEFT:"blue", RIGHT:"red", NEUTRAL:"gray"}
    trialToVisualise = 0
    # create png
    visualiser = AnimationCreator(STORAGE, myName, title, mapping,
                              trial=trialToVisualise, delay=30)
    # create animated gif
    visualiser.create_gif(verbose=True)


# run main
if __name__ == '__main__':
    main()
