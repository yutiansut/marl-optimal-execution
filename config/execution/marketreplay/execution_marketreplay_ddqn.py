import argparse
import datetime as dt
import pickle
import sys
import time

import numpy as np
import pandas as pd
from tensorflow import keras

from agent.examples.MomentumAgent import MomentumAgent
from agent.NoiseAgent import NoiseAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.ValueAgent import ValueAgent

from agent.examples.MarketReplayAgent import MarketReplayAgent
from agent.ExchangeAgent import ExchangeAgent
from agent.execution.baselines.twap_agent import TWAPExecutionAgent
from agent.execution.qlearning.ddqlearning_execution_agent import DDQLearningExecutionAgent
from Kernel import Kernel
from util import util
from util.model.QNets import *
from util.order import LimitOrder

sys.path.append(r"/efs/_abides/dev/mm/abides-dev/")

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

simulation_start_time = dt.datetime.now()

parser = argparse.ArgumentParser(description="Detailed options for market replay with execution agents config.")

parser.add_argument("-c", "--config", required=True, help="Name of config file to execute")
parser.add_argument("-s", "--seed", type=int, default=None, help="numpy.random.seed() for simulation")

parser.add_argument("-t", "--ticker", required=True, help="Name of the stock/symbol")
parser.add_argument("-d", "--date", required=True, help="Historical date")

parser.add_argument("--direction", default=None, help="BUY or SELL")
parser.add_argument("--parent_qty", type=int, default=None, help="Total Size of the parent order")
parser.add_argument("--start_hour", type=int, default=None, help="Time (in hours) to start executing the parent order")
parser.add_argument(
    "--horizon_length", type=int, default=None, help="Length of the execution time horizon (in minutes)"
)
parser.add_argument("--freq", default=None, help="Frequency of Order Placement")

parser.add_argument("-m", "--mode", default=None, help="train neural networks or test")
parser.add_argument("-a", "--agent_type", default=None, help="baseline or rl agent")

# parser.add_argument("-q", "--q_table", default=None, help="Q Table file")

parser.add_argument(
    "-l", "--log_dir", default=None, help="Log directory name (default: unix timestamp at program start)"
)
parser.add_argument("-lvl", "--level", default="1", help="Level of the orderbook")
parser.add_argument("--code", default="", help="Formulation code")
parser.add_argument("-v", "--verbose", action="store_true", help="Maximum verbosity!")

parser.add_argument("--config_help", action="store_true", help="Print argument options for this config file")

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

seed = args.seed  # Random seed specification on the command line.
if not seed:
    seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

FORMULATION_CODE = args.code
print(FORMULATION_CODE)

print("Mode: {}".format(args.mode))
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}".format(seed))
print("Symbol: {}".format(args.ticker))
print("Date: {}".format(args.date))

######################## Agents Config #########################################################################

# symbol
symbol = args.ticker

# Historical date to simulate.
historical_date = args.date
historical_date_pd = pd.to_datetime(historical_date)
mkt_open = historical_date_pd + pd.to_timedelta("09:30:00")
# Modified original 16:00:00
mkt_close = historical_date_pd + pd.to_timedelta("16:00:00")

agent_count, agents, agent_types = 0, [], []

num_agents = 10
agent_seeds = [np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))] * num_agents

# 1) Exchange Agent
agents.extend(
    [
        ExchangeAgent(
            id=0,
            name="EXCHANGE_AGENT",
            type="ExchangeAgent",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[symbol],
            pipeline_delay=0,
            computation_delay=0,
            stream_history=10,
            book_freq=None,  # set to 0 if orderbook needed
            log_events=True,
            log_orders=True,
            random_state=util.get_rand_obj(agent_seeds[0]),
        )
    ]
)
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Market Replay Agent
level = int(args.level)
file_name = f"{symbol}_{historical_date}_34200000_57600000_message_{level}.csv"
orders_file_path = f"data/lobster/LOBSTER_SampleFile_{symbol}_{level}/{file_name}"

agents.extend(
    [
        MarketReplayAgent(
            id=1,
            name="MARKET_REPLAY_AGENT",
            type="MarketReplayAgent",
            symbol=symbol,
            date=historical_date_pd,
            start_time=mkt_open,
            end_time=mkt_close,
            orders_file_path=orders_file_path,
            processed_orders_folder_path=f"data/marketreplay/level_{level}/",
            starting_cash=0,
            log_events=True,
            log_orders=True,
            random_state=util.get_rand_obj(agent_seeds[1]),
        )
    ]
)
agent_types.extend("MarketReplayAgent")
agent_count += 1


# 3) Momentum Agents
num_momentum_agents = 7
agents.extend(
    [
        MomentumAgent(
            id=j,
            name="MOMENTUM_AGENT_{}".format(j),
            type="MomentumAgent",
            symbol=symbol,
            starting_cash=0,
            min_size=1,
            max_size=10,
            wake_up_freq="20s",
            log_events=True,
            log_orders=True,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
        )
        for j in range(agent_count, agent_count + num_momentum_agents)
    ]
)
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")

# 4) Noise Agents
# num_noise = 10
# historical_date = pd.to_datetime(historical_date)
# noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")
# noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
# agents.extend(
#     [
#         NoiseAgent(
#             id=j,
#             name="NoiseAgent {}".format(j),
#             type="NoiseAgent",
#             symbol=symbol,
#             starting_cash=0,
#             wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
#             log_orders=True,
#             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
#         )
#         for j in range(agent_count, agent_count + num_noise)
#     ]
# )
# agent_count += num_noise
# agent_types.extend(["NoiseAgent"])


random_state = util.get_rand_obj(agent_seeds[2])

direction = args.direction
parent_qty = args.parent_qty

start_time = pd.Timestamp(f"{args.date} {args.start_hour}:00:00")
end_time = pd.Timestamp(f"{args.date} {args.start_hour}:00:00") + pd.Timedelta(minutes=args.horizon_length)
freq = args.freq
execution_time_horizon = pd.date_range(start=start_time, end=end_time, freq=freq)

# 5) TWAP Agent

twap_agent = TWAPExecutionAgent(
    id=agent_count,
    name="TWAP_EXECUTION_AGENT",
    type="TWAPExecutionAgent",
    symbol=symbol,
    starting_cash=0,
    execution_time_horizon=execution_time_horizon,
    freq=freq,
    direction=direction,
    quantity=parent_qty,
    log_events=True if args.mode == "train" else True,
    log_orders=True if args.mode == "train" else True,
    trade=True,
    random_state=random_state,
)

execution_agents = [twap_agent]
agents.extend(execution_agents)
agent_types.extend("TWAPExecutionAgent")
agent_count += 1


if args.agent_type == "rl":
    # RL Execution Agent Config
    # def unpickle_q_table(file_name):
    #     with open(file_name, "rb") as fin:
    #         q_table = pickle.load(fin)
    #     return q_table
    #
    # q_table = unpickle_q_table(args.q_table) if args.mode == "test" else None

    # instantiate two neural nets
    num_actions = 24

    eval_model = EvalModel(num_actions)
    target_model = TargetModel(num_actions)
    if args.mode == "test":
        # TODO: get proper director from args
        eval_model.load_weights("./saved_models/{}/eval_model_weights".format(FORMULATION_CODE))
        target_model.load_weights("./saved_models/{}/target_model_weights".format(FORMULATION_CODE))
        print("Weights loaded to target and eval models.")

    if args.mode == "train":
        # TODO: get proper director from args
        try:
            eval_model.load_weights("./saved_models/{}/eval_model_weights".format(FORMULATION_CODE))
            target_model.load_weights("./saved_models/{}/target_model_weights".format(FORMULATION_CODE))
            print("Weights loaded to target and eval models.")
        except:
            print("No weights, starting from scratch")

    ddq_learning_agent = DDQLearningExecutionAgent(
        id=agent_count,
        name="DDQN_EXECUTION_AGENT",
        type="DDQExecutionAgent",
        symbol=symbol,
        starting_cash=0,
        execution_time_horizon=execution_time_horizon,
        freq=freq,
        direction=direction,
        quantity=parent_qty,
        mode=args.mode,
        log_events=True if args.mode == "train" else True,
        log_orders=True if args.mode == "train" else True,
        trade=True,
        random_state=random_state,
        eval_model=eval_model,
        target_model=target_model,
    )
    execution_agents = [ddq_learning_agent]
    agents.extend(execution_agents)
    agent_types.extend("RLTWAPExecutionAgent")
    agent_count += 1

elif args.agent_type == "baseline":

    twap_agent = TWAPExecutionAgent(
        id=agent_count,
        name="TWAP_EXECUTION_AGENT",
        type="TWAPExecutionAgent",
        symbol=symbol,
        starting_cash=0,
        execution_time_horizon=execution_time_horizon,
        freq=freq,
        direction=direction,
        quantity=parent_qty,
        log_events=True if args.mode == "train" else True,
        log_orders=True if args.mode == "train" else True,
        trade=True,
        random_state=random_state,
    )

    execution_agents = [twap_agent]
    agents.extend(execution_agents)
    agent_types.extend("TWAPExecutionAgent")
    agent_count += 1

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel(
    "Market Replay Kernel",
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
)

kernelStartTime = historical_date_pd
kernelStopTime = end_time + pd.Timedelta("10min")

defaultComputationDelay = 0
latency = np.zeros((agent_count, agent_count))
noise = [0.0]

kernel.runner(
    agents=agents,
    startTime=kernelStartTime,
    stopTime=kernelStopTime,
    agentLatency=latency,
    latencyNoise=noise,
    defaultComputationDelay=defaultComputationDelay,
    defaultLatency=0,
    oracle=None,
    log_dir=args.log_dir,
)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
if args.agent_type == "rl":
    # ddq_learning_agent.plot_cost(0, 400, True)
    # ddq_learning_agent.plot_cost(0, 400, False)
    # ddq_learning_agent.plot_price()
    # ddq_learning_agent.plot_qty(False)
    # ddq_learning_agent.plot_qty(True)
    # ddq_learning_agent.plot_reward()
    print(sum(ddq_learning_agent.step_reward_hist))

if args.agent_type == "rl" and args.mode == "train":
    eval_model.save_weights("./saved_models/{}/eval_model_weights".format(FORMULATION_CODE))
    target_model.save_weights("./saved_models/{}/target_model_weights".format(FORMULATION_CODE))
    print("Models saved")
