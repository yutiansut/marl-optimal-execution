# FinTech Capstone

This repository is used to share some code about FinTech Capstone.

## Index

1. [Dependencies](#dependencies)
2. [Installation](#installation)
3. [Launch](#launch)
4. [Code Structure](#code-structure)
5. [ABIDES ReadMe](#abides-readme)

## Dependencies

- [anaconda](https://www.anaconda.com/distribution/)
- [git](https://git-scm.com)
- Recommended IDE: [Visual Studio Code](https://code.visualstudio.com). Recommended extensions:
  - [Bracket Pair Colorizer](https://marketplace.visualstudio.com/items?itemName=CoenraadS.bracket-pair-colorizer)
  - [GitLens — Git supercharged](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
  - [indent-rainbow](https://marketplace.visualstudio.com/items?itemName=oderwat.indent-rainbow)
  - [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint)
  - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  - [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv)

## Installation

- Clone the repository:

```bash
git clone git@github.com:xxx.git && cd xxx
```

- Install a conda environment:

```bash
conda create -n fintech-capstone python=3.7.6
conda activate fintech-capstone
pip install -r requirements.txt
```

- Link your VSCode `python.pythonPath` (in `.vscode/settings.json`) to your conda `fintech-capstone` environment.

## Launch

### ABIDES

- Run simulations:

```bash
scripts/marketreplay.sh
scripts/rmsc.sh
scripts/sparse_zi_100.sh
scripts/sparse_zi_1000.sh
scripts/impact_baseline.sh 3
scripts/impact_study.sh 3
```

- Run graphs:

```bash
scripts/graphs.sh
```

*N.B.:* If e.g. the execution of `scripts/graphs.sh` raises a `permission denied` error, execute:

```bash
chmod +x scripts/graphs.sh
```

### Realism metrics

*N.B.:* You may need to export the Python path of the fintech-capstone repository:

```bash
export PYTHONPATH="${PYTHONPATH}:/path_to_folder/fintech-capstone"
```

#### Asset return stylized facts

```bash
realism/plot_aamas2020_asset_return_stylized_facts.sh
```

#### Order flow stylized facts

```bash
realism/plot_order_flow_stylized_facts.sh
```

## Code Structure

The code structure presented in this boilerplate is grouped primarily by file type. Please note, however, that this structure is only meant to serve as a guide, it is by no means prescriptive.

```bash
tree -d -I '__pycache__|node_modules|graphs|log' > tree.txt
```

```bash
.
├── agent
│   ├── etf
│   ├── examples
│   ├── execution
│   └── market_makers
├── cli
├── config
├── contributed_traders
├── data
│   ├── 1m_ohlc
│   │   └── 1m_ohlc_2014
│   └── trades
│       └── trades_2014
├── message
├── model
├── realism
│   ├── metrics
│   └── plot_configs
│       └── plot_configs
│           ├── multiday
│           └── single_day
├── scripts
├── tests
└── util
    ├── formatting
    ├── model
    ├── oracle
    ├── order
    │   └── etf
    └── plotting
```

## ABIDES ReadMe

> ABIDES is an Agent-Based Interactive Discrete Event Simulation environment. ABIDES is designed from the ground up to support AI agent research in market applications. While simulations are certainly available within trading firms for their own internal use, there are no broadly available high-fidelity market simulation environments. We hope that the availability of such a platform will facilitate AI research in this important area. ABIDES currently enables the simulation of tens of thousands of trading agents interacting with an exchange agent to facilitate transactions. It supports configurable pairwise network latencies between each individual agent as well as the exchange. Our simulator's message-based design is modeled after NASDAQ's published equity trading protocols ITCH and OUCH.

Please see our arXiv paper for preliminary documentation:

<https://arxiv.org/abs/1904.12066>

Please see the wiki for tutorials and example configurations:

<https://github.com/abides-sim/abides/wiki>

## Quickstart

```bash
mkdir project
cd project

git clone https://github.com/abides-sim/abides.git
cd abides
pip install -r requirements.txt
```

## Run impact visualizations

- in `event_midpoint.py` and `event_ticker.py` files, replace `sim_files = sys.argv[2]` with `sim_files = sys.argv[2].split('\n')`
- Run 10 impact studies
- Run following commands:

```bash
python cli/event_midpoint.py GOOG "$(ls -at log/impact_*/Exchange*)"
python cli/event_ticker.py GOOG "$(ls -at log/impact_*/Exchange*)"
python cli/midpoint_plot.py GOOG log/impact_10/ExchangeAgent0.bz2
python cli/quote_plot.py GOOG log/impact_10/ExchangeAgent0.bz2
python cli/ticker_plot.py GOOG log/impact_10/ExchangeAgent0.bz2
```

## Run Marketreplay Q-Learning

- `scripts/execution/marketreplay/marketreplay.sh` (however you don't need to run it as I uploaded the generated logs on GitHub, this commands uses more than 16GB of RAM when run)
- `scripts/execution/marketreplay/marketreplay_rl_train.sh` (generates a lot of files in log/execution/marketreplay, don't know how to deal with the number of files generated for now and how to stop without interrupted execution brutally)
- `scripts/execution/qlearning.sh` computed the Q-Table and saves it in `log/execution/marketreplay` (you may need to reduce `num_episodes` if you didn't let the previous command execute enough episodes)

## Run DDQN Training

```bash
chmod +x scripts/execution/marketreplay/marketreplay_ddqn_train.sh
scripts/execution/marketreplay/marketreplay_ddqn_train.sh
```

## Run other scripts in ABIDES

```bash
scripts/rmsc02.sh
scripts/sparse_zi_100.sh
python -u abides.py -c sum -l sum -s 123456789
python -u abides.py -c value_noise -l value_noise -s 123456789
python -u abides.py -c qlearning -l qlearning -s 123456789
```

## Export Python Path
```bash
export PYTHONPATH="${PYTHONPATH}:/path_to_folder/marl-optimal-execution"
```