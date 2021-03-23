import argparse
import datetime as dt
import os
from multiprocessing import Pool

import numpy as np
import psutil


def run_in_parallel(
    config,
    num_simulations,
    num_parallel,
    mode,
    agent_type,
    ticker,
    # q_table,
    log_dir,
    formulation_code,
    seed,
    verbose,
):

    # dates = [
    #     "2003-01-{:02d}".format(day) for day in (
    #         list(range(21, 25)) + list(range(27, 32))
    #     )
    # ] + [
    #     "2003-02-{:02d}".format(day) for day in (
    #         list(range(3, 8)) + list(range(10, 15)) + list(range(17, 22)) + list(range(25, 29))
    #     )
    # ] + [
    #     "2003-03-{:02d}".format(day) for day in (
    #         list(range(3, 8)) + list(range(10, 15)) + list(range(17, 22)) + list(range(24, 29))
    #     )
    # ]
    # train_dates = dates[:24]
    # test_dates = dates[24:]

    dates = [
        "2003-01-{:02d}".format(day) for day in (list(range(13, 18)) + list(range(21, 25)) + list(range(27, 32)))
    ] + ["2003-02-{:02d}".format(day) for day in list(range(3, 7))]
    train_dates = dates[:9]
    test_dates = dates[9:]

    # train_dates = test_dates

    processes = None

    if mode == "train":
        directions = ["BUY"]
        # parent_quantities = [6e3, 6e3, 9e3, 12e3]
        # start_hours = [10, 11, 12, 13]
        # horizon_lengths = [30, 60, 90, 120]
        parent_quantities = [5e5]
        start_hours = [10]
        horizon_lengths = [330]  # all the period except first and last 30 minutes
        freq = "30S"

        processes = [
            f"python -u abides.py -c {config} -s {seed} -t {ticker} -d {date} "
            f"--direction {direction} --parent_qty {int(parent_qty)} --start_hour {start_hour} "
            f"--horizon_length {horizon_length} --freq {freq} "
            # f'-m {mode} -a {agent_type} {f"-q {q_table}" if q_table else ""} '
            f"-m {mode} -a {agent_type} "
            f"-l {log_dir}_{formulation_code}_{mode}_{ticker}_{date}_{direction}_{start_hour}_{horizon_length}_{int(parent_qty)} "
            f"--code {formulation_code} "
            f'{" -v" if verbose else ""}'
            for direction in directions
            for parent_qty in parent_quantities
            for date in train_dates
            for start_hour in start_hours
            for horizon_length in horizon_lengths
            for _ in range(1, num_simulations + 1)
        ]
        print(processes[0])

    elif mode == "test":
        directions = ["BUY"]
        parent_quantities = [5e5]
        start_hours = [10]
        horizon_lengths = [330]  # all the period except first and last 30 minutes
        freq = "30S"

        processes = [
            f"python -u abides.py -c {config} -s {seed} -t {ticker} -d {date} "
            f"--direction {direction} --parent_qty {int(parent_qty)} --start_hour {start_hour} "
            f"--horizon_length {horizon_length} --freq {freq} "
            # f'-m {mode} -a {agent_type} {f"-q {q_table}" if q_table else ""} '
            f"-m {mode} -a {agent_type} "
            f"-l {log_dir}_{formulation_code}_{mode}_{ticker}_{date}_{direction}_{start_hour}_{horizon_length}_{int(parent_qty)} "
            f"--code {formulation_code} "
            f'{" -v" if verbose else ""}'
            for direction in directions
            for parent_qty in parent_quantities
            for date in test_dates
            for start_hour in start_hours
            for horizon_length in horizon_lengths
            for _ in range(1, num_simulations + 1)
        ]
        print(processes[0])

        # ticker = "MMM"
        # date = "20200628"
        #
        # direction = "BUY"
        # parent_qty = 4e4
        #
        # start_hour = 10
        # horizon_length = 350
        # freq = "10S"
        #
        # processes = [
        #     f"python -u abides.py -c {config} -s {seed} -t {ticker} -d {date} "
        #     f"--direction {direction} --parent_qty {int(parent_qty)} --start_hour {start_hour} "
        #     f"--horizon_length {horizon_length} --freq {freq} "
        #     # f'-m {mode} -a {agent_type} {f"-q {q_table}" if q_table else ""} '
        #     f"-m {mode} -a {agent_type} "
        #     f"-l {log_dir}_{mode}_{ticker}_{date}_{direction}_{start_hour}_{horizon_length}_{int(parent_qty)} "
        #     f'{" -v" if verbose else ""}'
        # ]

    pool = Pool(processes=num_parallel)
    pool.map(run_process, processes)


def run_process(process):
    os.system(process)


if __name__ == "__main__":
    start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description="Main config to run multiple ABIDES simulations in parallel")

    parser.add_argument("--config", required=True, help="Name of config file to execute")
    parser.add_argument("--seed", type=int, required=False, help="Name of config file to execute")
    parser.add_argument("--num_simulations", type=int, default=1, help="Total number of simulations to run")
    parser.add_argument("--num_parallel", type=int, default=None, help="Number of processes to run in parallel")

    parser.add_argument("--mode", default=None, help="train (collect experiences) or test (use the Q-table)")
    parser.add_argument("--agent_type", default=None, help="baseline or rl agent")

    parser.add_argument("--ticker", required=True, help="Name of the stock/symbol")

    # parser.add_argument("--q_table", default=None, help="Q Table file if running in test mode")

    parser.add_argument("--log_dir", default=None, help="Log directory name (default: unix timestamp at program start)")
    parser.add_argument("--code", default="", help="Formulation code")
    parser.add_argument("-v", "--verbose", action="store_true", help="Maximum verbosity!")

    args, remaining_args = parser.parse_known_args()

    formulation_code = args.code
    config = args.config
    seed = args.seed
    num_simulations = args.num_simulations
    num_parallel = args.num_parallel if args.num_parallel else psutil.cpu_count()

    mode = args.mode
    agent_type = args.agent_type

    ticker = args.ticker

    # q_table = args.q_table

    log_dir = args.log_dir
    verbose = args.verbose

    np.random.seed(seed)

    print(f"Configuration: {config}")
    print(f"Seed: {seed}")
    print(f"Total number of simulation: {num_simulations}")
    print(f"Number of simulations to run in parallel: {num_parallel}")
    print(f"Mode: {mode}")
    print(f"Agent Type: {agent_type}")
    print(f"Ticker: {ticker}")

    run_in_parallel(
        config=config,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        mode=mode,
        agent_type=agent_type,
        ticker=ticker,
        # q_table=q_table,
        log_dir=log_dir,
        formulation_code=formulation_code,
        seed=seed,
        verbose=verbose,
    )

    end_time = dt.datetime.now()
    print(f"Total time taken to run in parallel: {end_time - start_time}")
