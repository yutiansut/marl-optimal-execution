# rm cache/*\.\.log*.pickle
# rm -rf visualizations
# python realism/asset_returns_stylized_facts.py -r data/1m_ohlc/1m_ohlc_2014/ -s log/random_fund_value -s log/random_fund_diverse  # -s log/hist_fund_value -s log/hist_fund_diverse
python realism/asset_returns_stylized_facts.py -r data/1m_ohlc/1m_ohlc_2019/ -s log/rmsc02 -s log/sparse_zi_100 -z
python realism/asset_returns_stylized_facts.py -r data/1m_ohlc/1m_ohlc_2019/ -s log/execution/marketreplay/marketreplay_rl_state_0_actions_0_buy_train_GOOG_20120621_BUY_10_350_40000 -z