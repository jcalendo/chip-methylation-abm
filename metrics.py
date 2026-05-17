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
    first_agent = next(iter(agentset))
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

def population_mean_locus_variance(agentset):
    """
    Measures intra-population heterogeneity by averaging the variance 
    of every individual CpG site across the population.
    """
    if len(agentset) < 2:
        return np.nan

    # Shape: (n_cells, n_genes, n_cpgs)
    population = np.array([a.cpgs for a in agentset])
    
    # Calculate variance along the cell axis (axis=0)
    # Shape becomes: (n_genes, n_cpgs)
    locus_variances = np.var(population, axis=0, ddof=1)
    
    # Return the global average of these variances
    return np.mean(locus_variances)

def population_discordance_fraction(agentset, lower_bound=0.2, upper_bound=0.8):
    """
    Measures the fraction of genes across the population that exhibit 
    discordant (intermediate) methylation patterns.
    """
    if len(agentset) == 0:
        return np.nan

    population = np.array([a.cpgs for a in agentset])
    
    # Calculate methylation fraction per gene per cell
    n_cpgs = population.shape[2]
    gene_meth_fractions = np.sum(population, axis=2) / n_cpgs
    
    # Identify discordant genes (between the bounds)
    is_discordant = (gene_meth_fractions > lower_bound) & (gene_meth_fractions < upper_bound)
    
    return np.mean(is_discordant)

def population_epigenetic_burden(agentset, threshold=0.10):
    """
    Measures the fraction of the population that exceeds a defined 
    mean methylation threshold. 
    """
    if len(agentset) == 0:
        return np.nan

    first_agent = next(iter(agentset))
    baseline = first_agent.init_meth 
    
    population = np.array([a.cpgs for a in agentset])
    cell_means = np.mean(population, axis=(1, 2))
    
    # Calculate fraction of cells exceeding baseline + threshold
    outlier_count = np.sum(cell_means > (baseline + threshold))
    
    return outlier_count / len(agentset)

def population_gini(agentset):
    """
    Computes the Gini coefficient of mean cell methylation across the population.
    Utilizes a sorted array calculation to maintain O(N log N) time complexity.
    """
    if len(agentset) < 2:
        return np.nan

    population = np.array([a.cpgs for a in agentset])
    
    # Get the "wealth" (mean methylation) of each cell
    cell_means = np.mean(population, axis=(1, 2))
    
    # Fast Gini calculation
    sorted_means = np.sort(cell_means)
    n = len(sorted_means)
    total_meth = np.sum(sorted_means)
    
    # Prevent division by zero if the entire population is completely unmethylated
    if total_meth == 0:
        return 0.0
        
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_means)) / (n * total_meth)
    
    return gini

def population_alpha_simpson(agentset, num_bins=10):
    """
    Computes Simpson's Diversity Index (1 - D) for the population.
    Requires binning the continuous methylation means into discrete 'species'.
    """
    if len(agentset) < 2:
        return np.nan

    population = np.array([a.cpgs for a in agentset])
    cell_means = np.mean(population, axis=(1, 2))
    
    # Bin the cells into discrete 'species' based on their methylation burden
    counts, _ = np.histogram(cell_means, bins=num_bins, range=(0.0, 1.0))
    
    # Calculate proportions (p_i)
    proportions = counts / len(agentset)
    
    # Simpson's Index D = sum(p_i^2)
    simpson_d = np.sum(proportions ** 2)
    
    # Return Gini-Simpson Index (1 - D) where higher = more diverse
    return 1.0 - simpson_d

def population_beta_bray_curtis(agentset, num_bins=10):
    """
    Measures the Bray-Curtis dissimilarity between the current population 
    and a theoretical Day-0 baseline population.
    """
    if len(agentset) < 2:
        return np.nan
        
    # Get the baseline state from the first agent in the set
    first_agent = next(iter(agentset))
    baseline_meth = first_agent.init_meth

    population = np.array([a.cpgs for a in agentset])
    cell_means = np.mean(population, axis=(1, 2))
    
    # Current population distribution
    current_counts, _ = np.histogram(cell_means, bins=num_bins, range=(0.0, 1.0))
    current_freqs = current_counts / len(agentset)
    
    # Theoretical baseline distribution (all cells clustered at init_meth)
    baseline_counts = np.zeros(num_bins)
    baseline_bin_index = int(baseline_meth * (num_bins - 1))
    baseline_counts[baseline_bin_index] = 1.0 
    
    # Bray-Curtis calculation: sum(|A - B|) / sum(A + B)
    numerator = np.sum(np.abs(current_freqs - baseline_counts))
    denominator = np.sum(current_freqs + baseline_counts)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator