## Agent based model of stochastic methylation during clonal hematopoiesis of indeterminate potential (chip)

**Work-in-progress** experimenting with methylation heterogeneity metrics and ABMs. 

The ABM simulates a constant size population of `Cells` and `Clones` after an initial growth phase. Each `Cell`/`Clone` is an agent with `n_genes` and `n_cpgs` per gene modeled as a 2D numpy array. Every CpG has a 'p_meth' independent probability of randomly gaining methylation (being flipped from a 0 -> 1) and an independent probability of becoming unmethylated if already methylated ('p_unmeth'). A single `Cell` is initialized at the start of the simulation. This `Cell` duplicates and then methylates/unmethylates until the target population size is reached. Following the growth phase, `Cell`s progress each step methylating/unmethylating stochastically until CH occurs. At a given age ('chip_time'), one of the `Cell`s is randomly chosen to become a `Clone` and one or more `Clone`s is initialized, replacing randomly selected `Cell`s to maintain a constant population size. At every step after, the `Clone`s can divide one of three mutually exclusive ways; duplication, death, or neutral division (no net change). NOTE: the probability of neutral division is computed by 'p_neutral' = 1.0 - ('p_duplicate' + 'p_die') and not given explicitly. 

For the division types, s = 'p_duplicate' - 'p_die' determines proliferation of `Clone`s in the population. If s > 0 then `Clone`s will proliferate faster. If s = 0 then the state of the simulation does not change with regard to the composition of the agents, if s < 0 then `Clone`s tend to die off. To maintain a constant population size after the growth phase, division is strictly coupled with replacement. When a `Clone` duplicates, a randomly chosen normal `Cell` is removed. Conversely, when a `Clone` dies, a randomly chosen normal `Cell` divides to fill the niche. Thus, the population size throughout the simulation stays constant after the initial growth phase. 

By default, the simulation is repeated 100 times. The data from all runs of the simulation is collected into a compressed .csv file and some simple plots are created showing the methylation and population dynamics.

## Set up and run

Install [uv](https://docs.astral.sh/uv/) then:

```
# Clone the repo
git clone https://github.com/jcalendo/chip-methylation-abm.git
cd chip-methylation-abm

# Sync project dependencies with `uv`
uv sync

# Run simulation script and view arguments
uv run main.py --help
```

## Usage

```
usage: main.py [-h] [--runs RUNS] [--n-proc N_PROC] [--n-agents N_AGENTS] [--steps STEPS] 
                [--chip-time CHIP_TIME] [--prop-chip PROP_CHIP] [--n-genes N_GENES] 
                [--n-cpgs N_CPGS] [--p-meth P_METH] [--p-unmeth P_UNMETH] [--init-meth INIT_METH] 
                [--p-meth-clone P_METH_CLONE] [--p-unmeth-clone P_UNMETH_CLONE] 
                [--p-duplicate P_DUPLICATE] [--p-die P_DIE] [--outfile OUTFILE] 
                [--quiet | --no-quiet]

Run the CHIP Methylation Agent-Based Model.

options:
  -h, --help            show this help message and exit
  --runs RUNS           Number of times to repeat the experiment. (default: 100)
  --n-proc N_PROC       Number of processes used for multiprocessing. (default: 4)
  --n-agents N_AGENTS   Total number of cells/agents. Constant population size after growth phase. (default: 1000)
  --steps STEPS         Total number of steps (years) to run the simulation. (default: 100)
  --chip-time CHIP_TIME
                        Step (year) at which the first mutant Clone is introduced. (default: 50)
  --prop-chip PROP_CHIP
                        Proportion of total population that will be turned into Clones at chip_time. (default: 0.02)
  --n-genes N_GENES     Number of genes (rows of CpG matrix) per agent. (default: 100)
  --n-cpgs N_CPGS       Number of CpGs (columns of CpG matrix) per gene. (default: 10)
  --p-meth P_METH       Probability of an unmethylated CpG becoming methylated per step. (default: 0.005)
  --p-unmeth P_UNMETH   Probability of a methylated CpG becoming unmethylated per step. (default: 1e-06)
  --init-meth INIT_METH
                        Initial proportion of CpGs that are methylated at the start of the simulation. (default: 0.0)
  --p-meth-clone P_METH_CLONE
                        Probability of an unmethylated CpG becoming methylated per step in Clones. (default: 0.005)
  --p-unmeth-clone P_UNMETH_CLONE
                        Probability of a methylated CpG becoming unmethylated per step in Clones. (default: 1e-06)
  --p-duplicate P_DUPLICATE
                        Probability of a Clone duplicating. (default: 0.0)
  --p-die P_DIE         Probability of a Clone being replaced by a Cell. (default: 0.0)
  --outfile OUTFILE     Filename to save run results DataFrame to. (default: run_results.csv.gz)
  --quiet, --no-quiet   Do not print program messages. (default: False)
```

## Examples

The current simulation has a lot of arguments to tweak. It's not immediately evident how adjusting these parameters leads to different behaviors. So I'll try to list a few examples to give an idea of what is currently possible.

### Simulating a constant 2% Clonal population size starting at CH onset

2% of Cells are removed at onset of CH and are replaced by exact copies of a selected Clone. These Clones persist until the end of the simulation. Their methylation still drifts but the composition of the population never changes. `--n-proc 12` & `--runs 100` indicates that we will repeat this experiment using the same parameters with different random numbers generator seeds 100 times using 12 parallel processes. 

```{bash}
uv run main.py --prop-chip 0.02 --p-duplicate 0.0 --p-die 0.0 --n-proc 12 --runs 100
```

<img src="docs/example1.png" width="800" alt="Plot showing constant 2% clonal population">

### Simulating a single Clone at CH onset with exponential Clonal growth

Setting `--prop-chip` to a value that results in a fractional number of Clones at onset will default to a single clone (e.g. 100 agents * 0.001 = 0.1 -> 1 Clone at onset). Once the Clone is initialized it will divide according to its selective advantage. Setting `--p-duplicate 1.0` and `p-die 0.0` ensures that every Clone will always divide into two Clones in the next step (i.e. exponential growth).

```{bash}
uv run main.py --n-agents 100 --prop-chip 1e-6 --p-duplicate 1.0 --p-die 0.0
```

<img src="docs/example2.png" width="800" alt="Plot showing exponential clonal growth">

### Simulating 1% Clones at age 40 with a strong selection advantage over Cells

The selection advantage (s) is given by `p_duplicate - p_die`. As stated above, if this value is positive then Clones have a fitness advantage over Cells. There are an infinite number of ways to tune these values to achieve the same value for `s`.

```{bash}
uv run main.py --chip-time 40 --prop-chip 0.01 --p-duplicate 0.25 --p-die 0.2
```

<img src="docs/example3.png" width="800" alt="Plot showing clones with selective advantage">

### Simulating different methylation drift/maintenance rates for Cells and Clones

The random chance of methylating or unmethylating a given CpG can be controlled for Cells and Clones independently. The initial methylation state of the founder Cell can also be set to simulate a founder Cell with a proportion methylation != 0.0.

```{bash}
uv run main.py \
  --init-meth 0.25 \
  --prop-chip 0.1 \
  --p-meth 0.001 \
  --p-unmeth 1e-6 \
  --p-meth-clone 0.01 \
  --p-unmeth-clone 0.001
```

<img src="docs/example4.png" width="800" alt="Plot showing different drift rates and starting values">

