import datetime

import jsons as js
import pandas as pd
import numpy as np

from agent.TradingAgent import TradingAgent
from agent.execution.baselines.execution_agent import ExecutionAgent
from ABIDESEnvMetrics import ABIDESEnvMetrics
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
        random_state=None,
        order_level = 1,
        a_q_map_steep_factor = 0.5):
        """
        add more memory placeholders for get_observation() method
        """
        super().__init__(
            id,
            name,
            type,
            symbol,
            starting_cash,
            direction,
            quantity,
            execution_time_horizon = execution_time_horizon,
            trade = trade,
            log_events = log_events,
            log_orders = log_orders,
            random_state = random_state)

        # new attributes added for dummy rl agent
        self.freq = freq
        # order_level is # of levels we want the agent to place order in.
        # This will determine the policy output size of the RL agent
        self.order_level = order_level
        # a hyperparameter that tunes how steep we want the action-to-volume mapping to be 
        self.a_q_map_steep_factor = a_q_map_steep_factor
        # effective_time_horizon is used for wakeup call schedule
        # execution_time_horizon attribute is for cancelorder schedule
        # this makes sure last cancelorder signal still within defined time range
        self.effective_time_horizon = execution_time_horizon[:-1]
        self.metrics = ABIDESEnvMetrics(maxlen = 100, quantity = quantity)
        self.rem_time = len(self.execution_time_horizon) - 1 # rem_time is in units of execution periods
        # self.rem_quantity from ExecutionAgent
        # self.accepted_orders = [] from ExecutionAgent

    def get_action_space_size(self):
        """
        get the action size which matches the self.order_level 
        e.g. when order_level = 1, the action has 2 values [total volume, level 1 %]
        e.g. when order_level = 2, the action has 3 values [total volume, level 1 %, level 2 %]
        """
        return self.order_level+1

    def process_action(self, action):
        """
        action: [total volume, level 1 info, level 2 info, level 3 info, ...]
        return:
        o: [level 1 vol, level 2 vol, level 3 vol, ...], order volumes
        """
        action = np.array(action).flatten()
        q0 = self.metrics.quantity              # total quantity
        q = self.metrics.rem_quantity           # remaining quantity
        x_hat = action[0]                       # total normalized volume to be placed
        if sum(action[1:]) == 0.0:
            o_hat = np.ones(len(action[1:]))/len(action[1:]) # equalize orders at all levels if outputs are all 0
        else:
            o_hat = action[1:]/sum(action[1:])  # normalized action in terms of %order to be placed at various levels
        q_hat = q/q0                            # normalized remaining quantity

        o_total = np.round(q0*q_hat*x_hat**(q_hat**self.a_q_map_steep_factor))   # total order volume
        o = np.round(o_total*o_hat)                                              # individual orders
        o[-1] = o_total - sum(o[0:-1])                                           # prevent errors from rounding; e.g. 3.5, 3.5 will round to 4,3
        return o

    def place_orders(self, currentTime, action):
        """
        loop over action and place orders

        action: [level 1 vol, level 2 vol, level 3 vol, ...], order volumes
        """
        orders = self.process_action(action)
        is_buy = True if self.direction == 'BUY' else False
        for l, q in enumerate(orders):
            try:
                # get price for current level
                bid, ask = self.metrics.getBidAskPrice(level = l+1, idx = 0)
                price = bid if is_buy else ask
                self.placeLimitOrder(symbol = self.symbol,
                                     quantity = q,
                                     is_buy_order = is_buy,
                                     limit_price = price,
                                     order_id=None,
                                     ignore_risk=True)
                log_print(f"[---- {self.name} - {currentTime} ----]: RL LIMIT ORDER PLACED - {q} @ {price}")
            except:
                # this situation happens when current LOB does not have sufficient level of bid or ask
                log_print(f"[---- {self.name} - {currentTime} ----]: RL LIMIT ORDER FAILED - {q} @ level {l+1}")


    def wakeup(self, currentTime):
        """
        Kernel calls dummy rl execution agent's wakeup method to trigger:
        1. schedule of next wakeup
        2. schedule of current order end time
        3. requesting LOB update from exchange agent
        """
        # Update agent???s currentTime
        # Handle first wake_up case
        # Handle market open
        can_trade = TradingAgent.wakeup(self, currentTime)
        # ----------- above actions handled by trading agent -------------
        if not can_trade:
            return
        # Schedule current endorder call using kernel???s setCancelOrder()
        if self.trade:
            try:
                # execution_time_horizon = pd.date_range(start=start_time, end=end_time, freq=freq)
                # use next wakeup time as current cancel order time, the CancelOrder signal
                # will be 0.5 ns earlier than next wakeup call
                self.setCancelOrder([time for time in self.execution_time_horizon if time > currentTime][0])
            except IndexError:
                log_print(f"[---- {self.name} -- {currentTime} ----]: RL Agent CancelOrder complete")
                self.trade = False
        # Schedule next wakeup call if time permits using setWakeup implemented in Agent through kernel???s setWakeup()
        if self.trade:
            try:
                self.setWakeup([time for time in self.effective_time_horizon if time > currentTime][0])
            except IndexError:
                log_print(f"[---- {self.name} -- {currentTime} ----]: RL Agent wakeups complete")
                self.trade = False

        # Call agent???s getCurrentSpread()* to receive LOB update
        self.getCurrentSpread(self.symbol, depth=500)
        self.state = "AWAITING_SPREAD"

        # # clean last period executed order memory
        # self.metrics.reset_curr_executed_orders()


    def setCancelOrder(self, requestedTime):
        """
        send EndOrder signal to kernel by calling kernel's setCancelOrder() method
        """
        self.kernel.setCancelOrder(self.id, requestedTime - pd.Timedelta(0.5))


    def receiveMessage(self, currentTime, msg):
        """

        """
        TradingAgent.receiveMessage(self, currentTime, msg)
        # handle order execution
        if msg.body["msg"] == "ORDER_EXECUTED":
            self.handleOrderExecution(currentTime, msg)
        # handle order acceptance
        elif msg.body["msg"] == "ORDER_ACCEPTED":
            self.handleOrderAcceptance(currentTime, msg)

        # handle QUERY_SPREAD
        if self.rem_quantity > 0 and self.state == "AWAITING_SPREAD" and msg.body["msg"] == "QUERY_SPREAD":
            self.state = "AWAITING_WAKEUP"
            self.metrics.addLOB(msg)

        # Note: cancel order and place order activities are moved into step process
        # 		"AWAITING_WAKEUP" state indicates agent has updated LOB info in this step
        # 		and is ready for place order


    def cancelAllOrders(self, currentTime=None):
        """used by the trading agent to cancel all of its orders."""
        if not currentTime:
            currentTime = self.currentTime
        for _, order in self.orders.items():
            log_print(f"[---- {self.name} - {currentTime} ----]: CANCELLED QUANTITY : {order.quantity}")
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
        # in super's method, self.executed_orders attribute is used to record full history of executed orders
        super().handleOrderExecution(currentTime, msg)
        # use self.cur_executed_orders to record current period executed orders
        # add more updates to agent's attributes per need
        self.metrics.update_rem_quantity(self.rem_quantity)

        executed_order = msg.body["order"]
        self.metrics.update_curr_executed_orders(executed_order)

    def get_remaining_time(self, current_time):
        curr_time = current_time.floor(self.freq)
        for i, t in enumerate(list(self.execution_time_horizon)):
            if t == curr_time:
                current_ts_index = i
                return len(self.execution_time_horizon) - 1 - current_ts_index
        return len(self.execution_time_horizon)

    def get_observation(self, currentTime):
        """
        compute and return state related features
        """
        rem_time = self.get_remaining_time(currentTime)
        self.rem_time = rem_time
        if not currentTime:
            currentTime = self.currentTime
        log_print(f'[---- {self.name} - {currentTime} ----]: observation calculated')
        obs = []
        obs += [self.rem_time, self.rem_quantity]
        obs.append(self.metrics.getLogReturn())
        obs.append(self.metrics.getBidAskSpread())
        obs.append(self.metrics.getVolImbalance())
        obs.append(self.metrics.getSmartPrice())
        obs.append(self.metrics.getMidPriceVolatility())
        obs.append(self.metrics.getTradeDirection())
        obs.append(self.metrics.getEffectiveSpread(level = 1, idx = 0))
        # obs.append(self.metrics.getPriceImprovement(idx = 0))
        # obs.append(self.metrics.getSharesFulfilledPercent())
        # obs.append(self.metrics.getPerMinuteVolRate())
        return np.array(obs)

    def get_observation_space_size(self):
        '''
        get the observation size, number is manually set to match the 
        output size of the method `get_observation(self, currentTime)`
        '''
        return 10


    def get_reward(self, currentTime):
        """
        compute reward for the action in current step
        """
        if not currentTime:
            currentTime = self.currentTime
        log_print(f'[---- {self.name} - {currentTime} ----]: reward calculated')
        # reward signal for macro price trend
        # need to get last two transacted price
        last_two_transacted_prices = self.metrics.getLastNPrice(n=2)
        # rem_quantity
        # current period total executed orders
        curr_executed_quantity = self.metrics.get_curr_executed_quantity()

        # reward signal for execution and placement strategy
        # latest arrival price (mid price)
        # fill price (weighted avg price of executed orders during current period)
        curr_total_cost = self.metrics.get_curr_executed_cost()
        try:
            curr_weighted_fill_price = curr_total_cost / curr_executed_quantity
        except ZeroDivisionError:
            curr_weighted_fill_price = 0

        # terminal reward comparing total cost with twap
        # full history of transacted price
        twap_mean_price = self.metrics.get_mean_historical_price()

        return None

    def clear_metrics_executed_order_buffer(self):
        self.metrics.reset_curr_executed_orders()


