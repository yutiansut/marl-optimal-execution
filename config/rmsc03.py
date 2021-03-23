# RMSC-3 (Reference Market Simulation Configuration) with data subscription mechanism:
# - 1     Exchange Agent
# - 1     Market Maker Agent
# - 50    Noise Agents
# - 10    Value Agents
# - 2    Momentum Agent

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd
from dateutil.parser import parse

from agent.examples.MomentumAgent import MomentumAgent
from agent.ExchangeAgent import ExchangeAgent
from agent.market_makers.POVMarketMakerAgent import POVMarketMakerAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from Kernel import Kernel
from util import util
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from util.order import LimitOrder

########################################################################################################################
############################################### GENERAL CONFIG #########################################################
# Config 2 - AAMAS Paper
parser = argparse.ArgumentParser(description="Detailed options for random_fund_value config.")
parser.add_argument("-c", "--config", required=True, help="Name of config file to execute")
parser.add_argument("-t", "--ticker", required=True, help="Ticker (symbol) to use for simulation")
parser.add_argument(
    "-d", "--historical-date", required=True, type=parse, help="historical date being simulated in format YYYYMMDD."
)
parser.add_argument(
    "-l", "--log_dir", default=None, help="Log directory name (default: unix timestamp at program start)"
)
parser.add_argument("-s", "--seed", type=int, default=None, help="numpy.random.seed() for simulation")
parser.add_argument("-v", "--verbose", action="store_true", help="Maximum verbosity!")
parser.add_argument("--config_help", action="store_true", help="Print argument options for this config file")
parser.add_argument("--mm-pov", type=float, default=0.05)
parser.add_argument("--mm-min-order-size", type=int, default=20)
parser.add_argument("--mm-window-size", type=int, default=5)
parser.add_argument("--mm-num-ticks", type=int, default=20)
parser.add_argument("--mm-wake-up-freq", type=str, default="1S")
parser.add_argument("--wide-book", action="store_true", help="Store orderbook in `wide` format")
args, remaining_args = parser.parse_known_args()
if args.config_help:
    parser.print_help()
    sys.exit()
log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed:
    seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)
util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose
log_orders = False
book_freq = 0
simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################
# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta("09:30:00")
mkt_close = historical_date + pd.to_timedelta("09:45:00")
agent_count, agents, agent_types = 0, [], []
# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.
r_bar = 1e5
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11
# Oracle
symbols = {
    symbol: {
        "r_bar": r_bar,
        "kappa": 1.67e-12,
        "agent_kappa": kappa,
        "sigma_s": 0,
        "fund_vol": 1e-4,
        "megashock_lambda_a": 2.77778e-13,
        "megashock_mean": 1e3,
        "megashock_var": 5e4,
        "random_state": np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
    }
}
oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)
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
            log_orders=True,
            pipeline_delay=0,
            computation_delay=0,
            stream_history=10,
            book_freq=book_freq,
            wide_book=args.wide_book,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
        )
    ]
)
agent_types.extend("ExchangeAgent")
agent_count += 1
# 2) Noise Agents
# TODO -- util.get_wake_time should have true open and close
num_noise = 50
noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")
noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
agents.extend(
    [
        NoiseAgent(
            id=j,
            name="NoiseAgent {}".format(j),
            type="NoiseAgent",
            symbol=symbol,
            starting_cash=starting_cash,
            wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
            log_orders=log_orders,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
        )
        for j in range(agent_count, agent_count + num_noise)
    ]
)
agent_count += num_noise
agent_types.extend(["NoiseAgent"])
# 3) Value Agents
num_value = 10
agents.extend(
    [
        ValueAgent(
            id=j,
            name="Value Agent {}".format(j),
            type="ValueAgent",
            symbol=symbol,
            starting_cash=starting_cash,
            sigma_n=sigma_n,
            r_bar=r_bar,
            kappa=kappa,
            lambda_a=lambda_a,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
        )
        for j in range(agent_count, agent_count + num_value)
    ]
)
agent_count += num_value
agent_types.extend(["ValueAgent"])
# 4) Market Maker Agent
num_mm_agents = 1
agents.extend(
    [
        POVMarketMakerAgent(
            id=j,
            name="POV_MARKET_MAKER_AGENT_{}".format(j),
            type="POVMarketMakerAgent",
            symbol=symbol,
            starting_cash=starting_cash,
            pov=args.mm_pov,
            min_order_size=args.mm_min_order_size,
            window_size=args.mm_window_size,
            num_ticks=args.mm_num_ticks,
            wake_up_freq=args.mm_wake_up_freq,
            log_orders=log_orders,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
        )
        for j in range(agent_count, agent_count + num_mm_agents)
    ]
)
agent_count += num_mm_agents
agent_types.extend("POVMarketMakerAgent")
# 5) Momentum Agents
num_momentum_agents = 2
agents.extend(
    [
        MomentumAgent(
            id=j,
            name="MOMENTUM_AGENT_{}".format(j),
            type="MomentumAgent",
            symbol=symbol,
            starting_cash=starting_cash,
            min_size=1,
            max_size=10,
            wake_up_freq="20s",
            log_orders=log_orders,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
        )
        for j in range(agent_count, agent_count + num_momentum_agents)
    ]
)
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")
# agents.extend([MomentumAgent(id=j,
#                              name="MOMENTUM_AGENT_{}".format(j),
#                              type="MomentumAgent",
#                              symbol=symbol,
#                              starting_cash=starting_cash,
#                              min_size=1,
#                              max_size=10,
#                              log_orders=log_orders,
#                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
#                                                                                        dtype='uint64')))
#                for j in range(agent_count, agent_count + num_momentum_agents)])
# agent_count += num_momentum_agents
# agent_types.extend("MomentumAgent")
########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################
kernel = Kernel(
    "Market Replay Kernel",
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
)
kernelStartTime = mkt_open
kernelStopTime = mkt_close + pd.to_timedelta("00:01:00")
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
    oracle=oracle,
    log_dir=args.log_dir,
)
simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
