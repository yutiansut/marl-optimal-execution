import datetime

import jsons as js
import pandas as pd

from agent.TradingAgent import TradingAgent
from agent.execution.baselines.execution_agent import ExecutionAgent
from util.util import log_print

# DummyRLAgent needs to handle train and test modes
# It needs to have task specific information since they are a part of observation and reward
# (presumably, external PPO algorithm does not need to know specific task)
# now allow agent to cancel reamining order when awaiting wakeup so that no need to seperate inventory and inventory allowed to trade

# functionalities:
# 1. wake up upon kernel request
# 2. get LOB update from exchange agent
# 3. calculate observation features
# 4. place order using action from kernel
# 5. track order acceptance and execution and update task specific parameters



# ### list of attributes from Agent -> FinancialAgent -> TradingAgent -> ExecutionAgent
	# Agent/FinancialAgent:
	# self.id = id
 #    self.name = name
 #    self.type = type
 #    self.random_state = random_state
 #    self.log_events = log_events
 #    self.kernel = None
 #    self.currentTime = None
 #    self.log = []

 #    TradingAgent:
 #    self.mkt_open = None
 #    self.mkt_close = None
 #    self.log_orders = log_orders
 #    self.starting_cash = starting_cash
 #    self.MKT_BUY = sys.maxsize
 #    self.MKT_SELL = 0
 #    self.holdings = {"CASH": starting_cash}
 #    self.orders = {}
 #    self.last_trade = {}
 #    self.daily_close_price = {}
 #    self.nav_diff = 0
 #    self.basket_size = 0
		# The agent remembers the last known bids and asks (with variable depth,
		# showing only aggregate volume at each price level) when it receives
   #      a response to QUERY_SPREAD.
 #    self.known_bids = {}
 #    self.known_asks = {}
 #    self.stream_history = {}
 #    self.transacted_volume = {}
 #    self.executed_orders = []
 #    self.first_wake = True
 #    self.mkt_closed = False
 #    self.book = ""

 #    ExecutionAgent:
 #    self.symbol = symbol
 #    self.direction = direction
 #    self.quantity = quantity
 #    self.execution_time_horizon = execution_time_horizon

 #    self.start_time = self.execution_time_horizon[0]
 #    self.end_time = self.execution_time_horizon[-1]
 #    self.schedule = None
 #    self.rem_quantity = quantity
 #    self.arrival_price = None

 #    self.state = "AWAITING_WAKEUP"
 #    self.accepted_orders = []
 #    self.trade = trade

class DummyRLExecutionAgent(ExecutionAgent):

	def __init__(
		self,
		id,
		name,
		type,
		symbol,
		starting_cash,
		direction,
		quantity,
		execution_time_horizon,
		freq,
		trade=True,
		log_events=False,
		log_orders=False,
		random_state=None):
		"""
		add more memory placeholders for get_observation() method
		"""
		super().__init__(
			id = id,
			name = name,
			type = type,
			symbol = symbol,
			starting_cash = starting_cash,
			direction = direction,
			quantity = quantity,
			execution_time_horizon = execution_time_horizon,
			trade = trade,
			log_events = log_events,
			log_orders = log_orders,
			random_state = random_state)

		# new attributes added for dummy rl agent
		self.freq = freq
		# effective_time_horizon is used for wakeup call schedule
		# execution_time_horizon attribute is for cancelorder schedule
		# this makes sure last cancelorder signal still within defined time range
		self.effective_time_horizon = execution_time_horizon[:-1]


	def wakeup(self, currentTime):
		"""
		Kernel calls dummy rl execution agent's wakeup method to trigger:
		1. schedule of next wakeup
		2. schedule of current order end time
		3. requesting LOB update from exchange agent
		"""
		# Update agent’s currentTime
		# Handle first wake_up case
		# Handle market open
		can_trade = super(TradingAgent, self).wakeup(currentTime)
		# ----------- above actions handled by trading agent -------------
		if not can_trade:
			return
		# Schedule current endorder call using kernel’s setCancelOrder()
		if self.trade:
			try:
				# execution_time_horizon = pd.date_range(start=start_time, end=end_time, freq=freq)
				# use next wakeup time as current cancel order time, the CancelOrder signal
				# will be 0.5 ns earlier than next wakeup call
				self.setCancelOrder([time for time in self.execution_time_horizon if time > current_time][0])
			except IndexError:
				log_print(f"[---- {self.name}  t={self.t} -- {current_time} ----]: RL Agent CancelOrder complete")
				self.trade = False
		# Schedule next wakeup call if time permits using setWakeup implemented in Agent through kernel’s setWakeup()
		if self.trade:
			try:
				self.setWakeup([time for time in self.effective_time_horizon if time > current_time][0])
			except IndexError:
				log_print(f"[---- {self.name}  t={self.t} -- {current_time} ----]: RL Agent wakeups complete")
				self.trade = False
		
		# Call agent’s getCurrentSpread()* to receive LOB update
		self.getCurrentSpread(self.symbol, depth=500)
		self.state = "AWAITING_SPREAD"


	def setCancelOrder(self, requestedTime):
		"""
		send EndOrder signal to kernel by calling kernel's setEndOrder() method 
		"""
		self.kernel.setEndOrder(self.id, requestedTime - pd.Timedelta(0.5))


	def receive_message(self, currentTime, msg):
		"""

		"""
		super(TradingAgent).receiveMessage(currentTime, msg)
		# handle order execution
		if msg.body["msg"] == "ORDER_EXECUTED":
			self.handleOrderExecution(currentTime, msg)
		# handle order acceptance
		elif msg.body["msg"] == "ORDER_ACCEPTED":
			self.handleOrderAcceptance(currentTime, msg)

		# handle QUERY_SPREAD
		if self.rem_quantity > 0 and self.state == "AWAITING_SPREAD" and msg.body["msg"] == "QUERY_SPREAD":
		    self.state = "AWAITING_WAKEUP"

		# Note: cancel order and place order activities are moved into step process
		# 		"AWAITING_WAKEUP" state indicates agent has updated LOB info in this step
		# 		and is ready for place order


	def cancelAllOrders(self, currentTime=None):
		"""used by the trading agent to cancel all of its orders."""
		if not currentTime:
			currentTime = self.currentTime
		for _, order in self.orders.items():
			log_print(f"[---- {self.name} t={self.t} - {currentTime} ----]: CANCELLED QUANTITY : {order.quantity}")
			if self.log_orders:
				self.logEvent("CANCEL_SUBMITTED", js.dump(order, strip_privates=True))
			self.cancelOrder(order)
			self.accepted_orders = []


	def handleOrderAcceptance(self, currentTime, msg):
		"""
		
		"""
		super().handleOrderAcceptance(currentTime, msg)
		# add more updates to agent's attributes per need


	def hanldeOrderExecution(self, currentTime, msg):
		"""
		
		"""
		super().handleOrderExecution(currentTime, msg)
		# add more updates to agent's attributes per need


	def get_observation(self, currentTime):
		"""
		compute and return state related features
		"""
		# TODO
		if not currentTime:
			currentTime = self.currentTime
		print(f'dummyRL Agent observation get at {currentTime}')
		return None




	def get_reward(self, currentTime):
		"""
		compute reward for the action in current step
		"""
		# TODO
		if not currentTime:
			currentTime = self.currentTime
		print(f'dummyRL Agent reward get at {currentTime}')
		return None
		

