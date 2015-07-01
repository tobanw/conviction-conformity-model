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
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import fsolve
from ComplexNetworkSim import NetworkAgent, Sim, NetworkSimulation, utils, PlotCreator, AnimationCreator

# output directory
STORAGE = 'results/test'

# Number of nodes 
NODES = 50

# Network topology
# TODO try other graphs, directed/weighted
G = nx.scale_free_graph(NODES)

# define public expressions, used as agent states
LEFT = -2.0/3
NEUTRAL = 0.0
RIGHT = 2.0/3

# initial states of agents
# TODO initial states should be irrelevant if I set the initial action to belief
states = [RIGHT for node in G.nodes()]
#states[0:25] = [LEFT]*25
#states[-25:] = [RIGHT]*25

# global params
THETA = (-0.7, -0.9, -1.89) # conviction, conformity, revision cost
GAMMA = (2,2)


### Agent behavior

class MyAgent(NetworkAgent):
    """ an implementation of an agent """

    def __init__(self, state, initialiser):
        NetworkAgent.__init__(self, state, initialiser) # doesn't accept stateVector

        self.theta = self.globalSharedParameters['theta']
        self.gamma = self.globalSharedParameters['gamma']
        # TODO try truncated normal instead of uniform distro of initial views
        self.stateVector = 2 * self.r.random() - 1 # private belief, uniform in [-1,1]

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
        maxima = np.array([ self.theta[0] * (a - self.updated_view(self.optim_lam(a)))**2
                            + self.theta[1] * self.ms_dev(a)
                            + self.theta[2] * self.optim_lam(a)**self.gamma[0]  
                            for a in acts ])
        globalMaxIndex = np.argmax(maxima)

        # set action
        self.state =acts[globalMaxIndex]

        # set belief
        self.stateVector = self.updated_view( self.optim_lam(acts[globalMaxIndex]) )

    def optim_lam(self, a):
        """
        solve non-linear foc for lambda, given discrete `a`
        """
        foc_lam = ( lambda lam: self.gamma[0] * self.theta[2]*lam**(self.gamma[0]-1)
                - 2 * self.theta[0] * (a - self.updated_view(lam) ) *
                (self.local_avg() - self.stateVector) )
        return fsolve( foc_lam, 0.5)[0]

    def updated_view(self, lam):
        return (1-lam)*self.stateVector + lam * self.local_avg()

    def local_avg(self):
        nbrs = self.getNeighbouringAgentsIter()
        # TODO edge weights (here is just using unweighted)
        return np.mean([nb.state for nb in nbrs])

    def ms_dev(self, a):
        nbrs = self.getNeighbouringAgentsIter()
        return np.mean([(a - nb.state)**2 for nb in nbrs])



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

# run main
if __name__ == '__main__':
    main()
