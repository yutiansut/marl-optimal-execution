import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


class Metrics:
    """Utility class to calculate metrics associated with the order book orders and snapshots.
    Attributes:
        orderbook_df: A pandas Dataframe describing the order book snapshots in time
        orders_df: A pandas Dataframe describing the orders stream
    """

    def __init__(self, symbol, date, orderbook_df, orders_df, bps=False):
        self.symbol = symbol
        self.date = date
        self.orderbook_df = orderbook_df
        self.orders_df = orders_df
        self.bps = bps

    def mid_price(self):
        """Returns the mid price in the form of a pandas series (Orderbook specific)"""
        mid_price = (self.orderbook_df.ask_price_1 + self.orderbook_df.bid_price_1) / 2
        return mid_price * 10000 if self.bps else mid_price

    def spread(self, type="naive"):
        """Returns the spread in the form of a pandas series (Orderbook specific)"""
        spread = None
        if type == "naive":
            spread = self.orderbook_df.ask_price_1 - self.orderbook_df.bid_price_1
        elif type == "effective":
            num_price_levels = 5
            volume_weighted_ask = sum(
                [
                    self.orderbook_df[f"ask_price_{level}"] * self.orderbook_df[f"ask_size_{level}"]
                    for level in range(1, num_price_levels + 1)
                ]
            ) / sum(self.orderbook_df[f"ask_size_{level}"] for level in range(1, num_price_levels + 1))
            volume_weighted_bid = sum(
                [
                    self.orderbook_df[f"bid_price_{level}"] * self.orderbook_df[f"bid_size_{level}"]
                    for level in range(1, num_price_levels + 1)
                ]
            ) / sum(self.orderbook_df[f"bid_size_{level}"] for level in range(1, num_price_levels + 1))
            spread = volume_weighted_ask.ffill() - volume_weighted_bid.ffill()
        return spread * 10000 if self.bps else spread

    def volume_order_imbalance(self):
        """Returns the volume order imbalance in the form of a pandas series (Orderbook specific)"""
        return self.orderbook_df.bid_size_1 / (self.orderbook_df.ask_size_1 + self.orderbook_df.bid_size_1)

    def order_flow_imbalance(self, tick_size=0.01, sampling_freq="1ms"):
        """Returns the order flow imbalance in the form of a pandas series (Orderbook specific)"""
        ofi_df = self.orderbook_df[["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"]].copy().reset_index()
        ofi_df["mid_price_change"] = ((ofi_df["bid_price_1"] + ofi_df["ask_price_1"]) / 2).diff().div(tick_size)
        ofi_df["PB_prev"] = ofi_df["bid_price_1"].shift()
        ofi_df["SB_prev"] = ofi_df["bid_size_1"].shift()
        ofi_df["PA_prev"] = ofi_df["ask_price_1"].shift()
        ofi_df["SA_prev"] = ofi_df["ask_size_1"].shift()
        ofi_df = ofi_df.dropna()
        bid_geq = ofi_df["bid_price_1"] >= ofi_df["PB_prev"]
        bid_leq = ofi_df["bid_price_1"] <= ofi_df["PB_prev"]
        ask_geq = ofi_df["ask_price_1"] >= ofi_df["PA_prev"]
        ask_leq = ofi_df["ask_price_1"] <= ofi_df["PA_prev"]
        ofi_df["OFI"] = pd.Series(np.zeros(len(ofi_df)))
        ofi_df["OFI"].loc[bid_geq] += ofi_df["bid_size_1"][bid_geq]
        ofi_df["OFI"].loc[bid_leq] -= ofi_df["SB_prev"][bid_leq]
        ofi_df["OFI"].loc[ask_geq] += ofi_df["SA_prev"][ask_geq]
        ofi_df["OFI"].loc[ask_leq] -= ofi_df["ask_size_1"][ask_leq]
        ofi_df = ofi_df.set_index("index")
        ofi_df = ofi_df[["mid_price_change", "OFI"]].resample(sampling_freq).sum().dropna()
        return ofi_df.OFI

    @staticmethod
    def twap(executed_trades_df):
        return executed_trades_df["fill_price"].cumsum() / pd.Series(
            np.arange(1, len(executed_trades_df) + 1), executed_trades_df.index
        )

    @staticmethod
    def vwap(executed_trades_df):
        return (executed_trades_df["fill_price"] * executed_trades_df["quantity"]).cumsum() / executed_trades_df[
            "quantity"
        ].cumsum()

    @staticmethod
    def slippage(exchange_executed_trades_df, agent_executed_trades_df, trade_direction, benchmark="VWAP", bps=True):
        direction = 1 if trade_direction == "BUY" else -1
        if benchmark == "VWAP":
            slippage = (
                direction
                * (Metrics.vwap(exchange_executed_trades_df) - Metrics.vwap(agent_executed_trades_df))
                / Metrics.vwap(exchange_executed_trades_df)
            )
        elif benchmark == "TWAP":
            slippage = (
                direction
                * (Metrics.twap(exchange_executed_trades_df) - Metrics.twap(agent_executed_trades_df))
                / Metrics.twap(exchange_executed_trades_df)
            )
        slippage = slippage[~slippage.isnull()]
        return slippage * 10000 if bps else slippage


class Plots:
    """Utility class to plot metrics associated with the order book orders and snapshots.
    Attributes:
        orderbook_df: A pandas Dataframe describing the order book snapshots in time
        orders_df: A pandas Dataframe describing the orders stream
    """

    def __init__(self, symbol, date, orderbook_df, orders_df, log_folder, bps=False):
        self.log_folder = log_folder
        self.metrics = Metrics(symbol, date, orderbook_df, orders_df, bps)

    def plot_mid_price(self, bps=False):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title(f"Mid Price ({'bps' if bps else '$'})", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Mid Price", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.mid_price())
        plt.savefig(self.log_folder + "/" + "Mid_Price.jpg", bbox_inches="tight")

    def plot_spread(self, bps=True):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title(f"Spread ({'bps' if bps else '$'})", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Spread", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.spread(type="naive"), label="Naive Spread")
        ax.plot(self.metrics.spread(type="effective"), label="Effective Spread")
        ax.legend()
        plt.savefig(self.log_folder + "/" + "Spread.jpg", bbox_inches="tight")

    def plot_volume_order_imbalance(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title("Volume Order Imbalance", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Volume Order Imbalance", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.volume_order_imbalance())
        plt.savefig(self.log_folder + "/" + "Volume_Order_Imbalance.jpg", bbox_inches="tight")

    def plot_order_flow_imbalance(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title("Order Flow Imbalance", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Order Flow Imbalance", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.order_flow_imbalance())
        plt.savefig(self.log_folder + "/" + "Order_Flow_Imbalance.jpg", bbox_inches="tight")


class Constants:
    POVExecutionWarning_msg = (
        "Running a configuration using POVExecutionAgent requires an ExchangeAgent with "
        "attribute `stream_history` set to a large value, recommended at sys.maxsize."
    )
