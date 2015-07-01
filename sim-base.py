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
from ComplexNetworkSim import NetworkAgent, Sim, NetworkSimulation

# Number of nodes 
NODES = 30

# Network topology
G = nx.scale_free_graph(NODES)

# define public expressions, used as agent states
LEFT = -1
NEUTRAL = 0
RIGHT = 1

# initial states of agents
# TODO initial states should be irrelevant if I set the initial action to belief
states = [NEUTRAL for node in G.nodes()]
states[0] = LEFT
states[1] = RIGHT

# global params
THETA = (-0.5, -0.5, -1.0)
GAMMA = (2,2)


### Agent behavior

class MyAgent(NetworkAgent):
    """ an implementation of an agent """

    def __init__(self, state, initialiser):
        NetworkAgent.__init__(self, state, initialiser) # doesn't accept stateVector

        self.lamb = 0.5
        # TODO try truncated normal instead of uniform
        self.stateVector = 2 * self.r.random() - 1 # private belief, uniform in [-1,1]

    def Run(self):
        while True:
            self.maximize()
            # wait 1 step before this agent gets run again
            yield Sim.hold, self, NetworkAgent.TIMESTEP_DEFAULT

    def maximize(self):
        """
        Solve for optimal action (which is self.state)
        """
        mc_conform = self.theta[1] * local_avg()
        # cts action:
        #acts = [(self.mc_conviction(b) + mc_conform)/(self.theta[0] +
        #    self.theta[1]) for b in (0,1)]
        acts = [LEFT,NEUTRAL,RIGHT]
        maxima = np.array([ [self.theta[0] * (a - self.mc_conviction(b))**2 +
                            self.theta[1] * self.ms_dev(a) +
                            self.theta[2] * b  
                            for b in (0,1) ] for a in acts])
        globalMaxIndex = np.unravel_index(np.argmax(maxima),maxima.shape)
        if globalMaxIndex[1] == 1: # agent updates view
            self.stateVector = self.updated_view()
        if globalMaxIndex[0] == LEFT:
            self.state = LEFT
        elif globalMaxIndex[0] == RIGHT:
            self.state = RIGHT
        else:
            self.state = NEUTRAL

    def mc_conviction(self, b):
        return self.theta[0] * (self.updated_view()*b +
                                self.stateVector*(1-b))

    def updated_view(self):
        return (1-self.lamb)*self.stateVector + self.lamb * self.local_avg()

    def local_avg(self):
        nbrs = self.getNeighbouringAgentsIter()
        # TODO edge weights (here is just using unweighted)
        return np.mean([nb.state for nb in nbrs])

    def ms_dev(self, a):
        nbrs = self.getNeighbouringAgentsIter()
        return np.mean([(a - nb.state)**2 for nb in nbrs])



### Simulation routine

# Simulation constants
MAX_SIMULATION_TIME = 20.0
TRIALS = 1

def main():
    directory = 'test' #output directory
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
