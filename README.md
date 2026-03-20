## Agent based model of stochastic methylation during clonal hematopoiesis of indeterminant potential (chip)

**Work-in-progress** experimenting with methylation heterogeneity metrics and ABMs. 

The ABM simulates a constant size population of Cells and Clones after an initial growth phase. Each Cell/Clone is an agent with n_genes and n_cpgs per gene modeled as a 2D numpy array. Every CpG has a 'p_meth' independent probability of randomly gaining methylation (being flipped from a 0 -> 1) and an independent probability of becoming unmethylated if already methylated (p_unmeth). A single Cell is initialized at the start of the simulation. This Cell duplicates and then (un)methylates until the target population size is reached. Following the growth phase, Cells progress each step (un)methylating stochastically until CH occurs. At a given age ('chip_time'), one of the Cells is chosen to become a Clone and one or more Clones is initialized, replacing randomly selected Cells to maintain a constant population size. At every step after, the Clones can divide one of three mutaully exclusive ways; duplication, death, or neutral division (no net change). NOTE: the probability of neutral division is computed by p_neutral = 1.0 - (p_duplicate + p_die) and not given explicitly. For the division types, s = p_duplicate - p_die determines proliferation of Clones in the population. If s > 0 then Clones will proliferate faster. If s = 0 then the state of the simulation does not change with regard to the composition of the agents, if s < 0 then Clones tend to die off. When a new Clone is created, a Cell is randomly chosen to be removed from the simulation. When a Clone dies it is replaced with a randomly chosen Cell from the population. Thus, the population size throughout the simulation stays constant after the initial growth phase. The default arguments are set to create exponential growth of clones (i.e. p_duplicate=1.0, p_die=0.0). By default, the simuation is repeated 100 times. The data from all runs of the simulation is collected into a compressed .csv file.

## TODO:

- Add error/bounds checking
- Add more metrics
- Parameter sweeps
- Induce CH at multiple times?
- Create plots in Python?
- How should we select the inital Clone? Randomly or by mean methylation?

## Set up and run

Install [uv](https://docs.astral.sh/uv/) then:

```
# Clone the repo
git clone git@github.com:jcalendo/chip-methylation-abm.git
cd chip-methylation-abm

# Sync project dependencies with `uv`
uv sync

# Run simulation script and view arguments
uv run main.py --help
```

## Examples

The current simulation has a lot of arguments to tweak. It's not immediately evident how adjusting these parameters leads to different behaviors. So I'll try to list a few examples to give an idea of what is currently possible.

### Simulating a constant 2% Clonal population size starting at CH onset

2% of Cells are removed at onset of CH and are replaced by exact copies of a selected Clone. These Clones persist until the end of the simulation. Their methylation still drifts but the composition of the population never changes. `--n-proc 12` & `--runs 100` indicates that we will repeat this experiment using the same parameters with different random numbers generator seeds 100 times using 12 parallel processess. 

```{bash}
uv run main.py --prop-chip 0.02 --p-duplicate 0.0 --p-die 0.0 --n-proc 12 --runs 100
```

### Simulating a single Clone at CH onset with exponential Clonal growth

Setting `--prop-chip` to a value that results in a fractional number of Clones at onset will default to a single clone (e.g. 100 agents * 0.001 = 0.1 -> 1 Clone at onset). Once the Clone is initialized it will divide according to its selective advantage. Setting `--p-duplicate 1.0` and `p-die 0.0` ensures that every Clone will always divide into two Clones in the next step (i.e. exponential growth).

```{bash}
uv run main.py --n-agents 100 --prop-chip 1e-6 --p-duplicate 1.0 --p-die 0.0
```

### Simulating 1% Clones at age 40 with a strong selection advantage over Cells

The selection advantage (s) is given by `p_duplicate - p_die`. As stated above, if this value is positive then Clones have a fitness advantage over Cells. There are an infinite number of ways to tune these values to acheive the same value for `s`. See below for more information.

```{bash}
uv run main.py --chip-time 40 --prop-chip 0.01 --p-duplicate 0.25 --p-die 0.2
```

### Simulating different methylation drift/maintanence rates for Cells and Clones

The random chance of methylating or unmethylating a given CpG can be controlled for Cells and Clones independently. The initial methylation state of a Cell can also be set to simulate a founder Cell with a proportion methylation > 0.0 (1% below).

```{bash}
uv run main.py --init-meth 0.01 --p-meth 0.01 --p-unmeth 0.001 --p-meth-clone 0.5 --p-unmeth-clone 0.01
```

## Defining selective advantage (s)

In a discrete-time model the growth of a mutant clone with a constant selective advantage `s` can be modeled using the discrete exponential growth equation:

$C_t = C_0 * (1 + s)^t$

Where:

* **C_t** is the target number of Clones.
* **C_0** is the initial number of Clones at induction.
* **t** is the number of simulation steps between induction and observation.
* **s** is the net selective advantage per step.

To determine the required fitness advantage to reach a specific variant allele frequency (VAF) or cellular fraction within a fixed timeframe, solve for `s`:

s = (C_target / C_0)^(1/t) - 1

In the fixed-population model, cellular turnover and clonal expansion are driven by 3 probabilities:

* **p_duplicate**: The probability an agent divides.
* **p_die**: The probability an agent is removed and replaced.
* **p_neutral**: The probability an agent remains inactive.

For a mutant clone to expand, its net growth advantage must equal `s`:

s = p_duplicate - p_die

Because the probabilities must sum to 1 (`p_duplicate + p_die + p_neutral = 1`), the exact values can be tuned based on the desired baseline cellular turnover rate (`p_neutral`). 

Note that in a stochastic model like this one, initializing one Clone results in a igh risk of early stochastic extinction (e.g., the agent rolls `p_die` on step 1), even with a strongly positive `s`.
