#!/bin/bash

obc_script=${PWD}/util/formatting/convert_order_book.py
osc_script=${PWD}/util/formatting/convert_order_stream.py

seed=123456789
ticker=GOOG
date=2012-06-21
nlevels=1

# Baseline (No Execution Agents)
baseline_log=marketreplay_level_${nlevels}/${ticker}_${date}
python -u abides.py -c marketreplay -t $ticker -d $date -lvl ${nlevels} -l ${baseline_log} -s $seed
python -u $obc_script ${PWD}/log/${baseline_log}/ORDERBOOK_${ticker}_FULL.bz2 ${ticker} ${nlevels} -o ${PWD}/log/${baseline_log}
python -u $osc_script ${PWD}/log/${baseline_log}/EXCHANGE_AGENT.bz2 ${ticker} ${nlevels} plot-scripts -o ${PWD}/log/${baseline_log}

# With Execution Agents
execution_log=execution_marketreplay_level_${nlevels}/${ticker}_${date}
python -u abides.py -c execution.marketreplay.execution_marketreplay -t $ticker -d $date -lvl ${nlevels} -l ${execution_log} -s $seed -e
python -u $obc_script ${PWD}/log/${execution_log}/ORDERBOOK_${ticker}_FULL.bz2 ${ticker} ${nlevels} -o ${PWD}/log/${execution_log}
python -u $osc_script ${PWD}/log/${execution_log}/EXCHANGE_AGENT.bz2 ${ticker} ${nlevels} plot-scripts -o ${PWD}/log/${execution_log}