#!/bin/bash

seed=123456789

config='execution.marketreplay.execution_marketreplay_ddqn'
log_dir='execution/marketreplay/marketreplay_ddqn_train'

formulation_code='FINAL'
mode='test'
agent_type='rl'

ticker='IBM'

num_simulations=1
num_parallel=1

python -u config/execution/marketreplay/execution_marketreplay_ddqn_parallel.py \
       --config ${config} \
       --seed ${seed} \
       --num_simulations ${num_simulations} \
       --num_parallel ${num_parallel} \
       --ticker ${ticker} \
       --mode ${mode} \
       --agent_type ${agent_type} \
       --log_dir ${log_dir} \
       --code ${formulation_code}