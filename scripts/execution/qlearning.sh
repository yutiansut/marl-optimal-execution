#!/bin/bash
log_folder='log/execution/marketreplay'
folder_name='marketreplay_rl_state_0_actions_0_buy'

python -u agent/execution/qlearning/qlearning_algo.py \
       --num_episodes 50  \
       --log_folder ${log_folder} \
       --folder_name ${folder_name}