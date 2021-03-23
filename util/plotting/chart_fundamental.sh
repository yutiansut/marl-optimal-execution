#!/usr/bin/env bash

python -u util/plotting/chart_fundamental.py -f log/rmsc02/fundamental_JPM.bz2 -f log/sparse_zi_100/fundamental_JPM.bz2 -l rmsc02 -l sparse_zi_100 -o visualizations
