from GymKernel import GymKernel
from agent_config import Agents
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
        self.seed = np.random.randint(low=0, high=2 ** 31 - 1) if seed == None else seed
        self.reset()

        # specify input and output shape
        RL_agent = self.agents.agent_list[self.agents.getAgentIndexByName('DUMMY_RL_EXECUTION_AGENT')[0]]
        action_space = ( [0.0] * RL_agent.get_action_space_size(), [1.0] * RL_agent.get_action_space_size() )
        observation_space = ( [0] * RL_agent.get_observation_space_size(), [0] * RL_agent.get_observation_space_size() )

        self.action_space = gym.spaces.Box(np.array(action_space[0]), np.array(action_space[1]) )
        self.observation_space = gym.spaces.Box(np.array(observation_space[0]), np.array(observation_space[1]) )



    def step(self, action):
        '''
        action is from an external agent
        '''
        # place order at the beginning of stepRunner() method
        reward, obs = self.kernel.stepRunner(action)

        # # move following operation into kernel's stepRunner method, then return observation and reward
        # RL_agent = self.agents.agent_list[self.agents.getAgentIndexByName('DummyRLExecutionAgent_name')]
        # obs = RL_agent.get_observation()
        # reward = RL_agent.compute_reward()

        if not self.kernel.messages.empty() and \
            self.kernel.currentTime and (self.kernel.currentTime <= self.kernel.stopTime):
            done = 0
        else:
            done = 1

        # TODO: the first step returned obs as [] while the rest of steps return None instead
        return obs, reward, done, None

    def reset(self):
        '''
        reinitialize kernel as agent information may be changed
        may be able to enable agent randomization between runs
        '''
        self.initAgents()
        self.initKernel()
        
    def initAgents(self, numMomentumAgent=5, numNoiseAgent=5):
        self.agents = Agents(self.ticker, self.date, seed=self.seed) # initialize the agents object for adding agents
        
        self.agents.addExchangeAgent()

        self.agents.addMarketReplayAgent()

        # self.agents.addTWAPExecutionAgent()

        self.agents.addDummyRLExecutionAgent()

        # for i in range(numMomentumAgent): 
        #     # note that individual agent parameterization may be needed to make this step meaningful
        #     self.agents.addMomentumAgent()

        # for i in range(numNoiseAgent):
        #     self.agents.addNoiseAgent()


    def initKernel(self):
        # by default, there should be only one dummy rl agent
        RL_agent = self.agents.agent_list[self.agents.getAgentIndexByName('DUMMY_RL_EXECUTION_AGENT')[0]]
        print(RL_agent)
        # pass entire Agents object in order to enable kernel of more advanced operations
        self.kernel = GymKernel("Market Replay Kernel",
                                random_state=np.random.RandomState(seed=self.seed-1),
                                RL_agent = RL_agent,
                                agents=self.agents.agent_list,)
        kernelStartTime = pd.to_datetime(self.date)
        kernelStopTime = pd.to_datetime(self.date) + pd.to_timedelta("16:10:00")
        defaultComputationDelay = 0
        latency = np.zeros((self.agents.num_agents, self.agents.num_agents))
        noise = [1.0]

        print(self.agents.getAgentIndexByName('DUMMY_RL_EXECUTION_AGENT'))


        self.kernel.initRunner(startTime=kernelStartTime,
                               stopTime=kernelStopTime,
                               agentLatency=latency,
                               latencyNoise=noise,
                               defaultComputationDelay=defaultComputationDelay,
                               defaultLatency=0,
                               oracle=None,
                               log_dir=self.log_dir)

if __name__ == "__main__":
    print("start")
    env = ABIDESEnv(ticker = "IBM", date = "2003-01-14", seed = 789)
    for i in range(12):
        # during the first 7 steps, query spread may not be called, so there's no get_observation() method call
        env.step(14)
