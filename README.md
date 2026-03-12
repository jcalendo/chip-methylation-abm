## Agent based model of stochastic methylation during clonal hematopoiesis of indeterminant potential (chip)

**Work-in-progress** experimenting with methylation heterogeneity metrics and ABMs. 

The ABM simulates a constant size population of Cells and Clones. Each Cell/Clone is an agent with n_genes and n_cpgs per gene modeled as a 2D numpy array. All CpGs are initialized as being completely unmethylated (all 0) at the start of the simulation. With each step of the simulation, every CpG has a p_meth probability of randomly gaining methylation (being flipped from a 0 -> 1) and an independent probability of becoming unmethylated if already methylated (p_unmeth). At a certain age (chip_time), one of the Cells is chosen to become a Clone. At every step after, the Clones can divide one of three mutaully exclusive ways; duplication, death, or neutral division. NOTE: the probability of neutral division is computed from p_neutral = 1.0 - (p_duplicate + p_die) and not given explicitly. For the division types, delta = p_duplicate - p_die determines proliferation of clones in the population. If delta > 0 then clones will proliferate faster. If delta = 0 then the state of the simulation does not change with regard to the agents, if delta < 0 then clones tend to die off. When a new Clone is created, a Cell is randomly chosen to be removed from the simulation. When a Clone dies it is replaced with a randomly chosen Cell from the population. Thus, the population size throughout the simulation stays constant. The default arguments are set to create exponential growth of clones (i.e. p_duplicate=1.0, p_die=0.0). During the simulation, heterogeniety metrics are computed for each agent and across the entire model. These metrics are saved as compressed .csv files.

### TODO:

- Check metrics for accuracy. Most were created as Gemini-converted-from-R-functions to numpy.
- Parameter sweeps
- Implement multiple simulation runs (replication of the same set of params)
- Induce CHIP at multiple times / induce multiple Clones
- Should Clones alter their methylation rate?

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
