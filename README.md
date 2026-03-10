## Agent based model of methylation during clonal hematopoiesis of indeterminant potential (chip)

Work-in-progress experimenting with methylation heterogeneity metrics and ABMs. 

The ABM simulates a constant size population of Cells and Clones. Each Cell/Clone is an agent with n_genes and n_cpgs per gene modeled as a 2D numpt array. All CpGs are initialized as being completely unmethylated (all 0) at the start of the simulation. With each step of the simulation, every CpG has a p_meth probability of randomly gaining methylation (being flipped from a 0 -> 1) and an independent probability of becoming unmethylated if already methylated (p_unmeth). At a certain age (chip_time), one of the Cells is randomly chosen to become a Clone. At every step after, the Clones can divide one of three mutaully exclusive ways; symmetric self renewal, symmetric differentiation, or asymmetric division. NOTE: the probability of asymetric division is computed from p_asym = 1.0 - (p_sym_renew + p_sym_diff) and not given explicitly. For the division types, delta = p_sym_renew - p_sym_diff determines the clone activity of the population. If delta > 0 then clones will proliferate faster. If delta = 0 then neutral drift occurs, if delta < 0 then clones tend to die off. When a new Clone is created by one of the division strategies a Cell is randomly chosen to be removed from the simulation. Thus, the population size throughout the simulation stays constant. During the simulation heterogeniety metrics are computed for each agent and across the entire model. These metrics are saved as compressed .csv files.

TODO:

- Figure out population level metrics. Added some Gemini-converted-from-R-functions but need to actually check them for correctness
- Create plots of metrics using computed results
- Parameter sweeps
- Implement multiple simulation runs (replication of the same set of params)
- Induce CHIP at multiple times induce multiple Clones
- Select Cells for CHIP induction (i.e. Pick the Cell with the highest methylation at chip_time)
- Determine if Clones should alter their methylation rate


### Set up and run

Install [uv](https://docs.astral.sh/uv/) then:

```
# Clone the repo
git clone git@github.com:jcalendo/chip-methylation-abm.git
cd chip-methylation-abm

# Sync project dependencies with `uv`
uv sync

# Run simulation script
uv run main.py
```
