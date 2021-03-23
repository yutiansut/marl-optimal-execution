import datetime
from collections import OrderedDict

import jsons as js
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD, RMSprop

from agent.execution.signals import mid_price, spread, volume_order_imbalance
from agent.execution.util import create_uniform_grid, discretize
from agent.TradingAgent import TradingAgent
from util.model.QNets import EvalModel, TargetModel
from util.util import log_print


class DDQLearningExecutionAgent(TradingAgent):
    # SIZE_ALLOCATION = {1: [1, 0, 0, 0], 2: [0.5, 0.5, 0, 0]}
    SIZE_ALLOCATION = {1: [1, 0, 0, 0], 2: [0.5, 0.5, 0, 0], 3: [0.34, 0.33, 0.33, 0]}
    SIZE_SCALE = [0.1] + [i * 0.5 for i in range(1, 6)]

    # SIZE_SCALE = [i * 0.25 for i in range(1, 11)]

    # construct action space mapping dict
    ACTIONS = dict()
    allocation_choice = list(SIZE_ALLOCATION.keys())
    allocation_choice.append(0)
    allocation_choice.sort()
    scale_choice = SIZE_SCALE
    action_name = 0
    for i in range(len(allocation_choice)):  # need to include market order choice
        for j in range(len(scale_choice)):
            # action name: (key for allocation choice, index for scale choice)
            ACTIONS[action_name] = (allocation_choice[i], scale_choice[j])
            action_name += 1

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
        n_features=6,
        n_actions=24,
        experience_size=64,
        replace_target_iter=5,
        batch_size=32,
        learning_rate=0.01,
        epsilon_increment=None,
        epsilon_max=0.9,
        reward_decay=0.98,
        eval_model=None,
        target_model=None,
        mode="train",
        trade=True,
        log_events=False,
        log_orders=False,
        random_state=None,
    ):

        super().__init__(
            id,
            name,
            type,
            starting_cash=starting_cash,
            log_events=log_events,
            log_orders=log_orders,
            random_state=random_state,
        )

        # client order information
        self.symbol = symbol
        self.direction = direction  # 'BUY' or 'SELL'
        self.quantity = quantity  # parent order size

        # order scheduler information
        self.execution_time_horizon = execution_time_horizon  # fixed frequency DatetimeIndex for the agent trade times
        self.start_time = self.execution_time_horizon[0]
        self.end_time = self.execution_time_horizon[-1]
        self.freq = freq
        self.schedule = self.generate_schedule()

        # deep q learning information
        self.n_features = n_features
        self.n_actions = n_actions  # ^^ 0: market order, 1-4: limit order according to size allocation
        self.train_step_counter = 0
        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon_increment = epsilon_increment
        self.epsilon_max = epsilon_max  # e greedy
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.reward_decay = reward_decay  # gamma

        self.remaining_time = len(self.execution_time_horizon) - 1  # remaining_time is in the units of execution time
        self.remaining_qty = quantity

        self.s = None
        self.a = None
        self.t = 0
        # neural nets
        self.eval_model = eval_model
        self.target_model = target_model
        self.eval_model.compile(
            # optimizer = SGD(learning_rate=0.1),
            optimizer=RMSprop(lr=self.learning_rate),
            loss="mse",
        )

        # ^^ change dictionary to ordered dictionary, to delete oldest history: od.popitem(last = False)
        self.experience = OrderedDict()  # ^^ Tuples of (s,a,s',r), s and s' are tuple as well
        self.experience_size = experience_size  # ^^ total of 450 experiences during 7.5 hr of trading

        # ^^ change dictionary to ordered dictionary, to delete oldest history: od.popitem(last = False)
        self.observation = OrderedDict()  # ^^ Tuples of (s,a,s',r), s and s' are tuple as well

        # ^^ use of this TBD
        self.discrete_state_grid = create_uniform_grid(low=[0, 0], high=[1.0, 1.0], bins=(200, 200))
        self.mode = mode
        self.state = "AWAITING_WAKEUP"
        self.trade = trade

        self.arrival_price = None
        self.accepted_orders = []
        self.cost_hist = []
        self.price_path = []
        self.action_hist = []
        self.step_reward_hist = []

        self.cost_hist_dict = OrderedDict()
        self.price_path_dict = OrderedDict()
        self.action_hist_dict = OrderedDict()
        self.step_reward_hist_dict = OrderedDict()

    def wakeup(self, current_time):
        can_trade = super().wakeup(current_time)
        if not can_trade:
            return
        if self.trade:
            try:
                self.setWakeup([time for time in self.execution_time_horizon if time > current_time][0])
            except IndexError:
                log_print(f"[---- {self.name}  t={self.t} -- {current_time} ----]: RL Agent wakeups complete")
                self.trade = False

        self.getCurrentSpread(self.symbol, depth=500)
        self.state = "AWAITING_SPREAD"

    def receiveMessage(self, current_time, msg):
        """[summary]

        Args:
            current_time ([type]): [description]
            msg ([type]): [description]
        """
        super().receiveMessage(current_time, msg)
        if msg.body["msg"] == "ORDER_ACCEPTED":
            # print("*order accepted*")
            self.handle_order_acceptance(current_time, msg)
        elif msg.body["msg"] == "ORDER_EXECUTED":
            # print("*order executed*")
            self.handle_order_execution(current_time, msg)
        elif (
            current_time in self.execution_time_horizon[:-1]
            and self.remaining_qty > 0
            and self.state == "AWAITING_SPREAD"
            and msg.body["msg"] == "QUERY_SPREAD"
        ):
            self.cancel_orders(current_time)
            # print("*order placed*")
            self.place_order(current_time)
            self.t += 1

    def kernelStopping(self):
        super().kernelStopping()

        slippage = (
            self.get_average_transaction_price() - self.arrival_price
            if self.direction == "BUY"
            else self.arrival_price - self.get_average_transaction_price()
        )

        self.logEvent("DIRECTION", self.direction, True)
        self.logEvent("TOTAL_QTY", self.quantity, True)
        self.logEvent("REM_QTY", self.remaining_qty, True)
        self.logEvent("ARRIVAL_MID", self.arrival_price, True)
        self.logEvent("AVG_TXN_PRICE", self.get_average_transaction_price(), True)
        self.logEvent("SLIPPAGE", slippage, True)

        agent_experience_df = pd.DataFrame(self.experience).T
        agent_experience_df.columns = ["s", "a", "s_prime", "r"]
        self.writeLog(dfLog=agent_experience_df, filename="agent_experience")

        agent_observation_df = pd.DataFrame(self.observation).T
        agent_observation_df.columns = [
            "time_remaining",
            "qty_remaining",
            "sprd",
            "vol_imbalance",
            "price_return",
            "return_from_start",
        ]
        self.writeLog(dfLog=agent_observation_df, filename="agent_observation")

        if self.mode == "train":
            cost_hist_df = pd.DataFrame(self.cost_hist_dict).T
            cost_hist_df.columns = ["cost_hist"]
            self.writeLog(dfLog=cost_hist_df, filename="agent_cost_hist")

        price_path_df = pd.DataFrame(self.price_path_dict).T
        price_path_df.columns = ["price_path"]
        self.writeLog(dfLog=price_path_df, filename="agent_price_path")

        action_hist_df = pd.DataFrame(self.action_hist_dict).T
        action_hist_df.columns = ["action_hist"]
        self.writeLog(dfLog=action_hist_df, filename="agent_action_hist")

        step_reward_hist_df = pd.DataFrame(self.step_reward_hist_dict).T
        step_reward_hist_df.columns = ["step_reward_hist"]
        self.writeLog(dfLog=step_reward_hist_df, filename="agent_step_reward_hist")

    def place_order(self, current_time):

        bids, asks = self.getKnownBidAsk(symbol=self.symbol, best=False)
        if current_time == self.start_time:
            self.arrival_price = mid_price(bids[0][0], asks[0][0])
            _, self.s = self.get_observation(current_time, bids, asks)
            log_print(f"[---- {self.name} t={self.t} -- {current_time} ----]: Arrival Mid Price {self.arrival_price}")
            log_print(
                f"[---- {self.name} t={self.t} -- {current_time} ----]: Parent Order Details: {self.direction} {self.quantity} "
                f"between {self.start_time} and {self.end_time}, {self.freq} frequency \n"
            )

        # ^^ currently don't know s' actually, so use same value as self.s for now; it will be updated next period
        observation, s_prime = self.get_observation(current_time, bids, asks)
        # choose action and place the order
        action_chosen = self.choose_action(s_prime)
        self.take_action(current_time, bids, asks, action_chosen)

        # create current experience entry
        # ^^ left reward as None for this period
        self.experience[self.t] = (self.s, action_chosen, s_prime, None)
        self.observation[self.t] = tuple(observation)
        # print("*experience updated*")
        # print(self.experience)

        # remove oldest experience when exceeds the experience_size
        # if len(self.experience.keys()) > self.experience_size:
        #     self.experience.popitem(last=False)

        # fit neural networks
        num_effective_experience = len(self.experience.keys()) - 1
        print("current period: {}".format(str(self.t)), end="\r")
        if (
            self.mode == "train"
            and self.remaining_qty > 0
            and num_effective_experience > self.batch_size
            and (self.train_step_counter % 5 == 0)
        ):
            # print("*training neural nets*", end='')
            self.train_neural_nets()

        self.s = s_prime
        self.a = action_chosen

        self.train_step_counter += 1

        # plot cost figure
        # if self.remaining_time == 1:
        #     self.plot_cost()

    def get_observation(self, current_time, bids, asks):
        """[summary]

        Args:
            current_time ([type]): [description]
            bids ([type]): [description]
            asks ([type]): [description]
        """
        def get_remaining_time(current_time):
            """[summary]

            Args:
                current_time ([type]): [description]

            Returns:
                [type]: [description]
            """
            current_time = current_time.floor(self.freq)
            for i, t in enumerate(list(self.execution_time_horizon)):
                if t == current_time:
                    current_ts_index = i
                    return len(self.execution_time_horizon) - 1 - current_ts_index
            return len(self.execution_time_horizon)

        # Private Variables (1) Time Remaining, (2) Quantity Remaining
        self.remaining_time = get_remaining_time(current_time)

        time_remaining = 2 * (self.remaining_time / len(self.execution_time_horizon)) - 1
        qty_remaining = 2 * (self.remaining_qty / self.quantity) - 1

        sprd = spread(best_bid_price=bids[0][0], best_ask_price=asks[0][0])
        vol_imbalance = volume_order_imbalance(best_bid_size=bids[0][1], best_ask_size=asks[0][1])

        mid_p = mid_price(bids[0][0], asks[0][0])
        self.price_path.append(mid_p)
        self.price_path_dict[self.t] = tuple([mid_p])

        if current_time == self.start_time:
            price_return = 0
        else:
            price_return = np.log(mid_p / self.price_path[-2])  # last one in the list is current mid price

        return_from_start = np.log(mid_p / self.price_path[0])

        # Market Variables:
        # (1) Spread: measure liquidity of current market, high liquidity could encourage limit order
        # (2) Volume Imbalance: indirectly reflect the price trend, influencing placement strategy on trendy price
        # (3) Volatility: risk measurement TBD
        # sprd = int(spread(best_bid_price=bids[0][0], best_ask_price=asks[0][0]))
        # vol_imbalance = round(volume_order_imbalance(best_bid_size=bids[0][1], best_ask_size=asks[0][1]), 2)
        # ^^ add market variables to observation TBD
        # observation = [time_remaining, qty_remaining]
        observation = [time_remaining, qty_remaining, sprd, vol_imbalance, price_return, return_from_start]
        # observation = np.round([time_remaining, qty_remaining, sprd, vol_imbalance, price_return, return_from_start], 4).tolist()
        discrete_observation = discretize(observation, self.discrete_state_grid)
        return observation, tuple(discrete_observation)

    def choose_action(self, s):
        """
        1. if train, epsilon greedy; if test, use neural nets to compute the best action

        :param s:
        :return:
        """
        a = None
        # prepare s as 2d array to feed into neural network
        s_array = np.array(s)
        s_array_2d = s_array[np.newaxis, :]
        # ^^ inside train mode, use epsilon greedy instead
        if self.mode == "train":
            # do not use exploitation until networks are trained for at least one time
            if np.random.uniform() < self.epsilon and len(self.experience) + 1 > self.batch_size:
                actions_value = self.eval_model.predict(s_array_2d)
                a = np.argmax(actions_value)
            else:
                a = self.random_state.randint(0, self.n_actions)

            # actions_value = self.eval_model.predict(s_array_2d)
            # a = np.argmax(actions_value)

        elif self.mode == "test":
            actions_value = self.eval_model.predict(s_array_2d)
            a = np.argmax(actions_value)

        return a

    def take_action(self, current_time, bids, asks, a):
        """
        1. place order based on 5 different kinds of actions

        :param current_time:
        :param bids:
        :param asks:
        :param s:
        :return:
        """
        qty = self.schedule[pd.Interval(current_time, current_time + datetime.timedelta(seconds=30))]
        action_space = DDQLearningExecutionAgent.ACTIONS
        allocation_type = action_space[a][0]

        if self.remaining_time == 1:
            qty = self.remaining_qty
            allocation_type = 0
        else:
            qty = max(0, round(action_space[a][1] * qty))

        self.action_hist.append(qty / self.quantity)
        self.action_hist_dict[self.t] = tuple([qty / self.quantity])

        if allocation_type == 0:
            # Action 1: Place a MARKET order
            log_print(f"[---- {self.name} t={self.t} -- {current_time} ----]: Placing a MARKET order for qty = {qty}")
            self.placeMarketOrder(symbol=self.symbol, direction=self.direction, quantity=qty)
        elif allocation_type in range(1, len(DDQLearningExecutionAgent.SIZE_ALLOCATION) + 1):
            # Action X: Place a LIMIT buy (sell) order at different bid (ask) price levels
            # price = bids[a][0] if self.direction == 'BUY' else asks[a][0]
            size_allocations = DDQLearningExecutionAgent.SIZE_ALLOCATION.get(allocation_type)
            for price_level, allocation in enumerate(size_allocations):
                size = round(allocation * qty)
                price = bids[price_level][0] if self.direction == "BUY" else asks[price_level][0]
                if size != 0:
                    log_print(
                        f"[---- {self.name} t={self.t} -- {current_time} ----]: "
                        f"Placing a LIMIT order for qty = {size} @ {price}, a={a}"
                    )
                    self.placeLimitOrder(
                        symbol=self.symbol, quantity=size, is_buy_order=self.direction == "BUY", limit_price=price
                    )
                    # print("*order placed*")

    def compute_reward(self, executed_order, current_time, s):
        r = None
        # 1) Reward based on slippage
        # if self.direction == "BUY":
        #     r = 1 - (
        #         (executed_order.fill_price - self.arrival_price) / self.arrival_price
        #     )  # ^^ arrival price is mid price
        # elif self.direction == "SELL":
        #     r = 1 - ((self.arrival_price - executed_order.fill_price) / self.arrival_price)

        if self.direction == "BUY":
            r = (
                (1 - ((executed_order.fill_price - self.arrival_price) / self.arrival_price))
                * executed_order.quantity
                / self.quantity
                * 10000
            )
            # ^^ arrival price is mid price
        elif self.direction == "SELL":
            r = (
                (1 - ((self.arrival_price - executed_order.fill_price) / self.arrival_price))
                * executed_order.quantity
                / self.quantity
                * 10000
            )

        # # 2) Reward modification based on s_prime
        # time_remaining = s[0]
        # qty_remaining = s[1]
        # T = len(self.execution_time_horizon)
        # trading_rate = qty_remaining / (T - time_remaining)
        # r = r * trading_rate  # ^^ early execution gets higher reward

        log_print(
            f"[---- {self.name} t={self.t} -- {current_time} ----]: "
            f"EXECUTED ORDER FILL PRICE: {executed_order.fill_price}, ARRIVAL PRICE: {self.arrival_price}, REWARD: {r}"
        )
        return r

    def train_neural_nets(self):
        """
        Fit two neural networks based on a batch of experience

        :return:
        """
        # convert experience OrderedDict to
        experience_lst = []
        full_lst = list(self.experience.items())
        full_lst.pop(-1)  # remove the latest experience since it's not complete yet
        # print(full_lst)
        for tpl in full_lst:
            temp = []
            for item in tpl[1]:  # index 1 is the value , index 0 is the key
                if type(item) == tuple:
                    temp.extend(list(item))
                else:
                    temp.append(item)
            experience_lst.append(temp)
        experience_array = np.array(experience_lst)
        # print(experience_array)

        # sample batch experience
        current_size = len(experience_array)
        current_batch_size = min(current_size, self.batch_size)
        sample_index = np.random.choice(current_size, current_batch_size)
        batch_experience = experience_array[sample_index, :]

        # print("*batch experience:*")
        # print(batch_experience)
        # print(type(batch_experience))
        # get q value from target_net and eval_net
        # if self.s:
        s_len = len(self.s)  # ^^ need to adjust self.n_features to represent the action size of s and s'
        # print("*confirm s_len*" + str(s_len))
        # target net uses s' data
        # print("*computing q_next*")
        q_next = self.target_model.predict(batch_experience[:, s_len + 1 : s_len * 2 + 1])
        # print("*q_next computed*")
        q_eval4next = self.target_model.predict(batch_experience[:, s_len + 1 : s_len * 2 + 1])
        # print("*q_eval4next computed*")
        # eval net uses s data
        q_eval = self.eval_model.predict(batch_experience[:, :s_len])  # possible try catch for len(None)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_experience[:, s_len].astype(
            int
        )  # ^^ get the action chosen from experience, action index is after length of self.s
        reward = batch_experience[:, -1]  # ^^ in our ndarray, the reward is the last column
        # get highest reward action from q_eval4next
        max_act4next = np.argmax(q_eval4next, axis=1)  # axis = 1 get max for each row
        # choose q_next based on the action chosen by q_eval4next
        selected_q_next = q_next[batch_index, max_act4next]
        q_target[batch_index, eval_act_index] = reward + self.reward_decay * selected_q_next

        # replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            # print("target net param updated")
        # train eval net
        cost = self.eval_model.train_on_batch(
            batch_experience[:, :s_len], q_target
        )  # ^^ cost be one of class fields TBD
        #  print("eval net trained")
        self.cost_hist.append(cost)
        self.cost_hist_dict[self.t] = tuple([cost])
        print(
            "current period: {}, cost history: {}{}".format(
                str(self.t), str(np.round(self.cost_hist[-min(len(self.cost_hist), 3) :], 5)), "                       "
            ),
            end="\r",
        )

        # update epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # update learn_step
        self.learn_step_counter += 1

    def getWakeFrequency(self):
        return self.start_time - self.mkt_open

    def generate_schedule(self):
        bins = pd.interval_range(start=self.start_time, end=self.end_time, freq=self.freq)
        schedule = {}
        child_quantity = int(self.quantity / (len(self.execution_time_horizon) - 1))
        for b in bins:
            schedule[b] = child_quantity
        log_print(f"[---- {self.name}  - Schedule ----]:")
        log_print(f"[---- {self.name}  - Total Number of Orders ----]: {len(schedule)}")
        for t, q in schedule.items():
            log_print(f"From: {t.left.time()}, To: {t.right.time()}, Quantity: {q}")
        return schedule

    def handle_order_execution(self, current_time, msg):
        """
        1. update remaining quantity
        2. fill current observation into s' of t-1 experience tuple
        3. update self.s current observation
        4. compute reward and fill it into t-1 experience tuple
        :param current_time:
        :param msg:
        :return:
        """
        executed_order = msg.body["order"]
        self.executed_orders.append(executed_order)
        total_executed_quantity = sum(executed_order.quantity for executed_order in self.executed_orders)
        # update remaining_qty
        self.remaining_qty = self.quantity - total_executed_quantity
        log_print(
            f"[---- {self.name} t={self.t} -- {current_time} ----]: LIMIT ORDER EXECUTED - {executed_order.quantity} @ {executed_order.fill_price}"
        )
        log_print(
            f"[---- {self.name} t={self.t} -- {current_time} ----]: TOTAL EXECUTED QUANTITY: {total_executed_quantity}, "
            f"REMAINING QUANTITY (NOT EXECUTED): {self.remaining_qty}, % EXECUTED: {round((1 - self.remaining_qty / self.quantity) * 100, 2)}"
        )

        if current_time.floor(self.freq) in self.execution_time_horizon:
            experience = list(self.experience[self.t - 1])  # ^^ get last period (t-1) experience
            bids, asks = self.getKnownBidAsk(symbol=self.symbol, best=False)
            s_prime_vals, experience[2] = self.get_observation(current_time, bids, asks)  # in t-1 experience, s' is now
            self.s = experience[2]

            r = self.compute_reward(executed_order, current_time, s_prime_vals)
            self.step_reward_hist.append(r)
            self.step_reward_hist_dict[self.t - 1] = tuple([r])
            # fill the reward into experience tuple
            experience[3] = r
            self.experience[self.t - 1] = tuple(experience)
            log_print(
                f"[---- {self.name} t={self.t} -- {current_time} ----]: "
                f"t={self.t - 1} -> s={self.experience[self.t - 1][0]}, a={self.experience[self.t - 1][1]}, "
                f"s_prime={self.experience[self.t - 1][2]}, r={self.experience[self.t - 1][3]} \n"
            )

    def handle_order_acceptance(self, current_time, msg):
        """
        1. update accepted order and accepted quantity
        2. fill current observation into s' of t-1 experience tuple
        3. accepted order has reward of -10 (it will be overrode if order is executed)

        :param current_time:
        :param msg:
        :return:
        """
        accepted_order = msg.body["order"]
        self.accepted_orders.append(accepted_order)
        accepted_qty = sum(accepted_order.quantity for accepted_order in self.accepted_orders)
        log_print(f"[---- {self.name} t={self.t} -- {current_time} ----]: ACCEPTED QUANTITY : {accepted_qty}")

        if current_time.floor(self.freq) in self.execution_time_horizon:
            experience = list(self.experience[self.t - 1])
            bids, asks = self.getKnownBidAsk(symbol=self.symbol, best=False)
            _, experience[2] = self.get_observation(current_time, bids, asks)
            self.s = experience[2]
            experience[3] = 0  # -10
            self.experience[self.t - 1] = tuple(experience)
            log_print(
                f"[---- {self.name} t={self.t} - {current_time} ----]: "
                f"t={self.t - 1} -> s={self.experience[self.t - 1][0]}, a={self.experience[self.t - 1][1]}, "
                f"s_prime={self.experience[self.t - 1][2]}, r={self.experience[self.t - 1][3]} \n"
            )

    def cancel_orders(self, current_time=None):
        """used by the trading agent to cancel all of its orders."""
        for _, order in self.orders.items():
            log_print(f"[---- {self.name} t={self.t} - {current_time} ----]: CANCELLED QUANTITY : {order.quantity}")
            if self.log_orders:
                self.logEvent("CANCEL_SUBMITTED", js.dump(order, strip_privates=True))
            self.cancelOrder(order)
            self.accepted_orders = []

    def plot_cost(self, start_index, end_index, limit):
        """"""
        # first several cost really unstable, exclude them for better plot
        start = max(0, start_index)
        end = min(end_index, len(self.cost_hist))
        data = self.cost_hist[start:end]
        if limit:
            plt.ylim(ymax=5)
        else:
            plt.ylim(ymax=100)
        plt.plot(np.arange(len(data)), data)
        plt.ylabel("Cost")
        plt.xlabel("Training Steps")
        plt.title("Double Deep Q Learning Agent Training Progress (step {} to {})".format(str(start), str(end)))
        plt.show()

    def plot_qty(self, include_last):
        """

        :return:
        """
        if include_last:
            data = self.action_hist
        else:
            data = self.action_hist[:-1]
        plt.plot(np.arange(len(data)), data)
        plt.ylabel("Percentage of Order Placed to Total Quantity")
        plt.xlabel("Time Steps")
        plt.title("DDQN Agent Order Placement Percentage during Training")
        plt.show()

    def plot_price(self):
        """

        :return:
        """
        data = self.price_path
        plt.plot(np.arange(len(data)), data)
        plt.ylabel("Price")
        plt.xlabel("Time Steps")
        plt.title("Price Path of Stock during Simulation")
        plt.show()

    def plot_reward(self):
        """

        :return:
        """
        data = self.step_reward_hist
        plt.plot(np.arange(len(data)), data)
        plt.ylabel("Reward")
        plt.xlabel("Time Steps")
        plt.title("Step Reward of DDQN during Training")
        plt.show()
