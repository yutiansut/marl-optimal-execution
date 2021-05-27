from ABIDESEnv.agent_config import agents
from GymKernel import GymKernel
import gym
import pandas as pd
import numpy as np

class ABIDESEnv(gym.Env):
    def __init__(self, ticker, date, log_dir=None, seed=None):
        '''
        ticker[str]: name of config file to execute
        date [str]: 'yyyy-mm-dd', historical date (may be able to randomize it later)
        seed[int]: numpy.random.seed() for simulation
        '''
        self.ticker = ticker
        self.date = date
        self.log_dir = log_dir
        self.seed = seed
        self.reset()

    def step(self, action):
        '''
        action is from an external agent
        '''
        self.kernel.stepRunner()

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        # incomplete
        RL_agent = self.agents.agent_list[self.agents.getAgentIndexByName('DummyRLExecutionAgent_name')]
        obs = RL_agent.get_observation()
        reward = RL_agent.compute_reward()
        if not self.kernel.messages.empty() and \
            self.kernel.currentTime and (self.kernel.currentTime <= self.kernel.stopTime):
            done = 0
        else:
            done = 1
            
        return obs, reward, done, None

    def reset(self):
        '''
        reinitialize kernel as agent information may be changed
        may be able to enable agent randomization between runs
        '''
        self.initAgents()
        self.initKernel()
        
    def initAgents(self, numMomentumAgent=5, numNoiseAgent=5):
        self.agents = agents(self.ticker, self.date, seed=self.seed) # initialize the agents object for adding agents
        
        self.agents.addExchangeAgent()

        self.agents.addMarketReplayAgent()

        for i in range(numMomentumAgent): 
            # note that individual agent parameterization may be needed to make this step meaningful
            self.agents.addMomentumAgent()

        for i in range(numNoiseAgent):
            self.agents.addNoiseAgent()

        self.agents.addTWAPExecutionAgent()
    
    def initKernel(self):
        self.kernel = GymKernel("Market Replay Kernel", random_state=np.random.RandomState(seed=self.seed-1))
        kernelStartTime = pd.to_datetime(self.date)
        kernelStopTime = pd.to_datetime(self.date) + pd.to_timedelta("16:10:00") 
        #??????????????? time formulation is a bit messy, kernel time, agent time (TWAP specifically) need to ask
        defaultComputationDelay = 0
        latency = np.zeros((self.agents.num_agents, self.agents.num_agents))
        noise = [1.0]

        self.kernel.initRunner(agents=self.agents.agent_list,
                               startTime=kernelStartTime,
                               stopTime=kernelStopTime,
                               agentLatency=latency,
                               latencyNoise=noise,
                               defaultComputationDelay=defaultComputationDelay,
                               defaultLatency=0,
                               oracle=None,
                               log_dir=self.log_dir)