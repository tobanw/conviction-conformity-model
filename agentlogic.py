"""
Agent class:
    - utility / cost function params
    - decision rule (focs)
    - attibutes: self-consciousness, 
    - state: private conviction
"""

import numpy as np
from scipy.optimize import fsolve
from ComplexNetworkSim import NetworkAgent, Sim

### Agent behavior

class MyAgent(NetworkAgent):
    """
    Network agent objects: contains agent behavior and states.
    Actions are buffered to get synchronous actions (best response dynamics).
    Actions are stored in `state`, views in `stateVector`.
    """

    def __init__(self, state, initialiser):
        NetworkAgent.__init__(self, state, initialiser) # doesn't accept stateVector

        self.theta = self.globalSharedParameters['theta']
        self.gamma = self.globalSharedParameters['gamma']
        self.acts = self.globalSharedParameters['acts']
        # TODO try truncated normal instead of uniform distro of initial views
        self.stateVector = 2 * self.r.random() - 1 # private belief, uniform in [-1,1]
        self.local_avg = 0.0 # initialize the attribute

        # start off agents acting according to their beliefs
        self.init_states()

        # initialize action buffer for synchronous updates
        self.buffered_action = self.state

    def init_states(self):
        n = len(self.acts)
        tmp_acts = self.acts[:]
        tmp_acts.append(1 + 1.0/n)
        partition = np.array(tmp_acts) - 1.0/n
        for k in range(n):
            if partition[k] <= self.stateVector < partition[k+1]:
                self.state = self.acts[k]

    def Run(self):
        while True:
            self.maximize()
            # wait 1 step before this agent gets run again
            yield Sim.hold, self, NetworkAgent.TIMESTEP_DEFAULT

    def maximize(self):
        """
        Solve for optimal action (which is self.state)
        """

        # get local_avg action once to economize on computation
        self.set_local_avg()

        maxima = np.array([ self.theta[0] * (a - self.updated_view(self.optim_lam(a)))**2
            + self.theta[1] * self.ms_dev(a)
            + self.theta[2] * self.optim_lam(a)**self.gamma[0] *
            self.polar_cost()  for a in self.acts ])
        globalMaxIndex = np.argmax(maxima)

        # set action
        self.buffered_action = self.acts[globalMaxIndex]

        # set belief
        self.stateVector = self.updated_view( self.optim_lam(self.acts[globalMaxIndex]) )

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
        return fsolve( foc_lam, 0.3)[0]

    def updated_view(self, lam):
        return (1-lam)*self.stateVector + lam * self.local_avg

    def set_local_avg(self):
        nbrs = self.getNeighbouringAgentsIter()
        # TODO edge weights (here is just using unweighted)
        try:
            self.local_avg = np.mean([nb.state for nb in nbrs])
        except RuntimeWarning:
            self.local_avg = self.state

    def ms_dev(self, a):
        nbrs = self.getNeighbouringAgentsIter()
        try:
            return np.mean([(a - nb.state)**2 for nb in nbrs])
        except RuntimeWarning:
            return 0

    def polar_cost(self):
        # changing views more costly when avg is far
        return abs(self.local_avg - self.stateVector)**self.gamma[1]

class Synchronizer(NetworkAgent):
    """
    Environment agent who implements action choices at end of each time step
    """

    def __init__(self, state, initialiser):
        NetworkAgent.__init__(self, 0, initialiser)

    def Run(self):
        while True:
            self.simultaneous_update()
            # wait 1 step before this agent gets run again
            yield Sim.hold, self, NetworkAgent.TIMESTEP_DEFAULT

    def simultaneous_update(self):
        for agent in self.getAllAgents():
            agent.state = agent.buffered_action
