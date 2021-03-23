mkdir data/plotting/level_1/
mkdir data/plotting/level_1/JPM/
mkdir data/plotting/level_1/JPM/rmsc02/
mkdir data/plotting/level_1/JPM/sparse_zi_100/
mkdir data/plotting/level_1/GOOG/
mkdir data/plotting/level_1/IBM/
mkdir data/plotting/level_1/GOOG/rl/
mkdir visualizations/order_flow/
mkdir visualizations/order_flow/JPM/
mkdir visualizations/order_flow/JPM/rmsc02/
mkdir visualizations/order_flow/JPM/sparse_zi_100/
mkdir visualizations/order_flow/GOOG/
mkdir visualizations/order_flow/IBM/
mkdir visualizations/order_flow/GOOG/rl/

python util/formatting/prepare_abides_data_for_plotting.py log/rmsc02/EXCHANGE_AGENT.bz2 JPM 20190628 -o data/plotting/level_1/JPM/rmsc02/
python realism/order_flow_stylized_facts.py data/plotting/level_1/JPM/rmsc02/ -o visualizations/order_flow/JPM/rmsc02/ -z

python util/formatting/prepare_abides_data_for_plotting.py log/sparse_zi_100/ExchangeAgent0.bz2 JPM 20190628 -o data/plotting/level_1/JPM/sparse_zi_100/
python realism/order_flow_stylized_facts.py data/plotting/level_1/JPM/sparse_zi_100/ -o visualizations/order_flow/JPM/sparse_zi_100/ -z

python util/formatting/prepare_abides_data_for_plotting.py log/marketreplay_level_1/GOOG_2012-06-21/EXCHANGE_AGENT.bz2 GOOG 20120621 -o data/plotting/level_1/GOOG/
python realism/order_flow_stylized_facts.py data/plotting/level_1/GOOG -o visualizations/order_flow/GOOG/ -z

python util/formatting/prepare_abides_data_for_plotting.py log/marketreplay_level_1/IBM_2003-01-13/EXCHANGE_AGENT.bz2 IBM 20120621 -o data/plotting/level_1/IBM/
python realism/order_flow_stylized_facts.py data/plotting/level_1/IBM -o visualizations/order_flow/IBM/ -z

python util/formatting/prepare_abides_data_for_plotting.py log/execution/marketreplay/marketreplay_ddqn_train_train_GOOG_20120621_BUY_10_350_40000/EXCHANGE_AGENT.bz2 GOOG 20120621 -o data/plotting/level_1/GOOG/rl/
python realism/order_flow_stylized_facts.py data/plotting/level_1/GOOG/rl/ -o visualizations/order_flow/GOOG/rl/ -z