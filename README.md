## Agent based model of stochastic methylation during clonal hematopoiesis of indeterminant potential (chip)

**Work-in-progress** experimenting with methylation heterogeneity metrics and ABMs. 

The ABM simulates a constant size population of Cells and Clones after an initial growth phase. Each Cell/Clone is an agent with n_genes and n_cpgs per gene modeled as a 2D numpy array. Every CpG has a 'p_meth' independent probability of randomly gaining methylation (being flipped from a 0 -> 1) and an independent probability of becoming unmethylated if already methylated (p_unmeth). A single Cell is initialized at the start of the simulation. This Cell duplicates and then (un)methylates until the target population size is reached. Following the growth phase, Cells progress each step (un)methylating stochastically until CH occurs. At a given age ('chip_time'), one of the Cells is chosen to become a Clone. At every step after, the Clones can divide one of three mutaully exclusive ways; duplication, death, or neutral division (no net change). NOTE: the probability of neutral division is computed by p_neutral = 1.0 - (p_duplicate + p_die) and not given explicitly. For the division types, delta = p_duplicate - p_die determines proliferation of Clones in the population. If delta > 0 then Clones will proliferate faster. If delta = 0 then the state of the simulation does not change with regard to the composition of the agents, if delta < 0 then Clones tend to die off. When a new Clone is created, a Cell is randomly chosen to be removed from the simulation. When a Clone dies it is replaced with a randomly chosen Cell from the population. Thus, the population size throughout the simulation stays constant after the initial growth phase. The default arguments are set to create exponential growth of clones (i.e. p_duplicate=1.0, p_die=0.0). By default, the simuation is repeated 100 times. The data from all runs of the simulation is collected into a compressed .csv file.

### TODO:

- Check metrics for accuracy. Most were created as Gemini-converted-from-R-functions to numpy.
- Parameter sweeps
- Induce CHIP at multiple times / induce multiple Clones

### Set up and run

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
