mkdir graphs

# book_plot.py only works for marketreplay
python cli/book_plot.py "$(ls -at log/marketreplay_level_1/GOOG_2012-06-21/ORDERBOOK* | head -n 1)"

python cli/sparse_fundamental.py log/sparse_zi_100/fundamental_JPM.bz2
python cli/sparse_midpoint.py JPM log/sparse_zi_100/Exchange*
python cli/sparse_ticker.py JPM log/sparse_zi_100/Exchange* log/sparse_zi_100/ZIAgent2*
