import numpy as np
from scipy.spatial.distance import jensenshannon


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


def mean_shannon(agent):
    """
    Compute the mean Shannon entropy of the genes.
    A gene with 50% methylation has high entropy (disordered).
    A gene with 0% or 100% methylation has 0 entropy (ordered).
    """
    # Get the fraction of methylation for each gene
    p_me = np.mean(agent.cpgs, axis=1)
    p_un = 1.0 - p_me

    with np.errstate(divide="ignore", invalid="ignore"):
        h = -1 * ((p_me * np.log2(p_me)) + (p_un * np.log2(p_un)))

    h = np.nan_to_num(h, nan=0.0)

    return np.mean(h)


def beta_var(agent):
    """
    Compute the variance of gene-level beta-values.
    Measures epigenetic heterogeneity across the genome.
    """
    gene_betas = np.mean(agent.cpgs, axis=1)
    return np.var(gene_betas, ddof=1) if len(gene_betas) > 1 else 0.0


def gini(agent):
    """
    Compute the Gini mean difference for gene-level beta-values.
    Measures epigenetic polarization (e.g., are genes either fully meth or fully unmeth?).
    """
    gene_betas = np.mean(agent.cpgs, axis=1)
    n = len(gene_betas)

    if n < 2:
        return 0.0

    sorted_betas = np.sort(gene_betas)
    weights = 2 * np.arange(1, n + 1) - n - 1
    gmd = 2 * np.sum(weights * sorted_betas) / (n**2)

    return gmd
