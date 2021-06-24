from agent.execution.rl.dummy_rl_execution_agent import DummyRLExecutionAgent
import numpy as np
import pandas as pd
from agent.Agent import Agent
from agent.examples.MomentumAgent import MomentumAgent
from agent.examples.MarketReplayAgent import MarketReplayAgent
from agent.ExchangeAgent import ExchangeAgent
from agent.execution.baselines.twap_agent import TWAPExecutionAgent
from agent.NoiseAgent import NoiseAgent
from util import util

class Agents():
    def __init__(self, symbol, date, seed=None):
        '''
        symbol [str]: name of the stock
        seed [int]: an integer for seeding agents (seed is incremented by 1 for different agents)
        date [str]: 'yyyy-mm-dd', historical date (may be able to randomize it later)
        '''
        self.symbol = symbol
        self.date = date
        self.date_pd = pd.to_datetime(self.date)
        self.seed = np.random.randint(low=0, high=2 ** 32) if seed == None else seed
        
        self.mkt_open = self.date_pd + pd.to_timedelta("09:30:00") # market open timestamp
        self.mkt_close = self.date_pd + pd.to_timedelta("16:00:00") # market close timestamp, modified original 16:00:00
        
        self.agent_list = []
        self.num_agents = 0

    def addAgent(self, agent): 
        '''
        agent [Agent]: an ABIDES Agent object, the id and random states will be overwritten
        '''
        if isinstance(agent, Agent):
            agent.id = self.num_agents
            agent.random_state = np.random.RandomState(seed=self.seed+self.num_agents)
            self.agent_list += [agent]
            self.num_agents += 1
        else:
            raise TypeError('input object does not inherit from ABIDES Agent class')

    def addExchangeAgent(self):
        agent = ExchangeAgent(id=self.num_agents,
                              name=f"{self.num_agents}_EXCHANGE_AGENT",
                              type="ExchangeAgent",
                              mkt_open=self.mkt_open,
                              mkt_close=self.mkt_close,
                              symbols=[self.symbol],
                              pipeline_delay=0, # why 0?
                              computation_delay=0, # why 0?
                              stream_history=10,
                              book_freq=None,  # set to 0 if orderbook needed
                              log_events=True,
                              log_orders=True,
                              random_state=np.random.RandomState(seed=self.seed+self.num_agents))
        self.agent_list += [agent]
        self.num_agents += 1

    def addMarketReplayAgent(self, level=1):
        '''
        level [int]: Level of the orderbook, for reading data set
        '''
        file_name = f"{self.symbol}_{self.date}_34200000_57600000_message_{level}.csv"
        orders_file_path = f"data/lobster/LOBSTER_SampleFile_{self.symbol}_{level}/{file_name}"

        agent = MarketReplayAgent(id=self.num_agents,
                                  name=f"{self.num_agents}_MARKET_REPLAY_AGENT",
                                  type="MarketReplayAgent",
                                  symbol=self.symbol,
                                  date=self.date_pd,
                                  start_time=self.mkt_open,
                                  end_time=self.mkt_close,
                                  orders_file_path=orders_file_path,
                                  processed_orders_folder_path=f"data/marketreplay/level_{level}/",
                                  starting_cash=0,
                                  log_events=True,
                                  log_orders=True,
                                  random_state=np.random.RandomState(seed=self.seed+self.num_agents))
        self.agent_list += [agent]
        self.num_agents += 1

    def addMomentumAgent(self, starting_cash=0, min_size=1, max_size=10, wake_up_freq="20s"):
        agent = MomentumAgent(id=self.num_agents,
                              name=f"{self.num_agents}_MOMENTUM_AGENT",
                              type="MomentumAgent",
                              symbol=self.symbol,
                              starting_cash=starting_cash, # why 0???????????????
                              min_size=min_size,
                              max_size=max_size,
                              wake_up_freq=wake_up_freq,
                              log_events=True,
                              log_orders=True,
                              random_state=np.random.RandomState(seed=self.seed+self.num_agents))
        self.agent_list += [agent]
        self.num_agents += 1

    def addNoiseAgent(self, starting_cash=0):
        noise_mkt_open = self.date_pd + pd.to_timedelta("09:00:00")
        noise_mkt_close = self.date_pd + pd.to_timedelta("16:00:00")
        agent = NoiseAgent(id=self.num_agents,
                           name=f"{self.num_agents}_NOISE_AGENT",
                           type="NoiseAgent",
                           symbol=self.symbol,
                           starting_cash=starting_cash,
                           wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                           log_orders=True,
                           random_state=np.random.RandomState(seed=self.seed+self.num_agents))
        self.agent_list += [agent]
        self.num_agents += 1

    def addTWAPExecutionAgent(self, direction="BUY", quantity=1e5, freq="30S", start_time="09:30:00", end_time="16:00:00",
                              starting_cash=0):
        start_t = self.date_pd + pd.to_timedelta(start_time)
        end_t = self.date_pd + pd.to_timedelta(end_time)
        execution_time_horizon = pd.date_range(start=start_t, end=end_t, freq=freq)
        agent = TWAPExecutionAgent(id=self.num_agents,
                                   name=f"{self.num_agents}_TWAP_EXECUTION_AGENT",
                                   type="TWAPExecutionAgent",
                                   symbol=self.symbol,
                                   starting_cash=starting_cash,
                                   execution_time_horizon=execution_time_horizon,
                                   freq=freq,
                                   direction=direction,
                                   quantity=quantity,
                                   log_events=True,
                                   log_orders=True,
                                   trade=True,
                                   random_state=np.random.RandomState(seed=self.seed+self.num_agents))
        self.agent_list += [agent]
        self.num_agents += 1

    def addDummyRLExecutionAgent(self, direction = "BUY", quantity=1e5, freq="30S", starting_cash=0, 
                                 start_time="09:30:00", end_time="16:00:00", log_events=False, log_orders=False):
        start_t = self.date_pd + pd.to_timedelta(start_time)
        end_t = self.date_pd + pd.to_timedelta(end_time)
        execution_time_horizon = pd.date_range(start=start_t, end=end_t, freq=freq)
        agent = DummyRLExecutionAgent(id = self.num_agents, 
                                      name = f"{self.num_agents}_DUMMY_RL_EXECUTION_AGENT",
                                      type = "DummyRLExecutionAgent",
                                      symbol = self.symbol,
                                      starting_cash = starting_cash,
                                      direction = direction,
                                      quantity = quantity,
                                      execution_time_horizon = execution_time_horizon,
                                      freq = freq,
                                      trade = True,
                                      log_events = log_events,
                                      log_orders = log_orders,
                                      random_state = np.random.RandomState(seed=self.seed+self.num_agents)
                                      )
        self.agent_list.append(agent)
        self.num_agents += 1

    def getAgentTypes(self):
        return [agent.type for agent in self.agent_list]
        
    def getAgentNames(self):
        return [agent.name for agent in self.agent_list]

    def getAgentIndexByName(self, lookup_name):
        '''
        name [str]: the index of the agent name
        '''
        names = self.getAgentNames()
        temp = []
        for name in names:
            if lookup_name in name:
                temp.append(name)

        if not temp:
            raise ValueError('input agent name does not exist')
        else:
            return [names.index(matched) for matched in temp]







# # Agent / FinancialAgent

# # id [int]: agent unique id (usually autoincremented)
# # Name [str]: for human consumption, should be unique (often type + number)
# # Type [??? ]: for machine aggregation of results, should be same for all agents following the same strategy 
# #              (incl. parameter settings).
# # random_state [np.random.RandomState]: random state to use for any agent stochastic needs
# # log_events [bool]: flag for enabling logging
# self.id = id
# self.name = name
# self.type = type
# self.random_state = random_state
# self.log_events = log_events




# # ExchangeAgent(FinancialAgent)

# # mkt_open [pandas._libs.tslibs.timestamps.Timestamp]: market open time 
# # mkt_close [pandas._libs.tslibs.timestamps.Timestamp]: market close time
# # symbols [list]: a list of stock names
# # book_freq [??????????????? int or str]: frequency of archiving orderbook for analysis 
# # wide_book [bool]: Store orderbook in wide format? ONLY WORKS with book_freq == 0
# # pipeline_delay [int]: a parallel processing pipeline delay (exchange agent specific) in nanosec. 
# #                       This is an additional delay added only to order activity (placing orders, etc) 
# #                       and not simple inquiries (market operating hours, etc)
# # computation_delay [int]: agent process time (exchange agent specific), usually set to 1ns unless for special perposes
# # stream_history [int??????????]: The exchange maintains an order stream of all orders leading to the last L trades
# #                          to support certain agents from the auction literature (GD, HBL, etc).

# mkt_open,
# mkt_close,
# symbols,
# book_freq="S",
# wide_book=False,
# pipeline_delay=40000,
# computation_delay=1,
# stream_history=0,





# # TradingAgent(FinancialAgent)

# # starting_cash [int]: in cents, agent's initial cash holding
# # log_orders [bool]: flag for enabling logging orders
# starting_cash=100000
# log_orders=False




# # ExecutionAgent(TradingAgent)

# # symbol [str]: name of the stock
# # direction [str]: BUY or SELL
# # quantity [int]: Total Size of the order
# # execution_time_horizon [pandas.core.indexes.datetimes.DatetimeIndex]: a series of timestaps for wakeup
# # trade [bool]: flag for enabling trade