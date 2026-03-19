import numpy as np

from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import jensenshannon


# Agent-level metrics -----------------------------------------------------------------------------


def mean_methylation(agent):
    """Global mean methylation across the entire genome (all genes and CpGs)."""
    return np.mean(agent.cpgs)


def pattern_freq(agent):
    """
    Compute the density distribution of gene methylation.
    Returns the frequency of genes that have 0, 1, 2... n methylated CpGs.
    """
    counts = np.sum(agent.cpgs, axis=1)  # Sum across CpGs for each gene
    return np.bincount(counts, minlength=agent.n_cpgs + 1) / agent.n_genes


def jsd_unmeth(agent):
    """
    Compute JSD using an 'unmeth' reference distribution.
    Measures how far the cell's gene density distribution is from a fully unmethylated state.
    """
    freqs = pattern_freq(agent)
    ref_vec = np.zeros(agent.n_cpgs + 1)
    ref_vec[0] = 1.0

    return jensenshannon(freqs, ref_vec) ** 2


def evenness(agent):
    """
    Compute Pielou's evenness index across gene methylation densities.
    """
    freqs = pattern_freq(agent)
    h = entropy(freqs)
    s = np.log(agent.n_cpgs + 1)

    return h / s


def emd_unmeth(agent):
    """
    Compute the Earth Mover's Distance (Wasserstein distance) between
    the cell's methylation density and a fully unmethylated reference.
    Unlike JSD, EMD accounts for the 'distance' between bins.
    """
    freqs = pattern_freq(agent)

    # The 'locations' of the bins on the x-axis (e.g., 0.0, 0.25, 0.5, 0.75, 1.0)
    bin_locations = np.linspace(0, 1, agent.n_cpgs + 1)

    # Reference distribution (100% of the weight is at the 0.0 bin)
    ref_freqs = np.zeros(agent.n_cpgs + 1)
    ref_freqs[0] = 1.0

    # Calculate EMD (requires the bin locations and their respective weights/frequencies)
    emd = wasserstein_distance(bin_locations, bin_locations, freqs, ref_freqs)

    return emd


# Population-level metrics -----------------------------------------------------------------------


def population_meth_mean(agentset):
    """Compute measurements across all genes"""
    if len(agentset) < 2:
        return np.nan

    # Cell x gene x cpg array
    population = np.array([a.cpgs for a in agentset])

    # Mean over the flattened array
    return np.mean(population)


def population_meth_var(agentset):
    """
    Measures the variance of the mean methylation across agents (cells)
    """
    if len(agentset) < 2:
        return np.nan

    population = np.array([a.cpgs for a in agentset])
    cell_means = np.mean(population, axis=(1, 2))

    return np.var(cell_means, ddof=1)


def population_jsd(agentset):
    """
    Compute the JSD relative to a completely unmethylated reference
    """
    if len(agentset) == 0:
        return np.nan

    # Just need one agent to get params
    first_agent = agentset[0]
    n_cpgs = first_agent.n_cpgs
    n_cells = len(agentset)

    population = np.array([a.cpgs for a in agentset])
    counts = np.sum(population, axis=2)
    bins = np.arange(n_cpgs + 1)
    bin_counts = (counts[:, :, None] == bins).sum(axis=0)
    freqs = bin_counts / n_cells

    reference = np.zeros((1, n_cpgs + 1))
    reference[0, 0] = 1.0
    jsds = jensenshannon(freqs, reference, axis=1) ** 2

    return np.mean(jsds)
