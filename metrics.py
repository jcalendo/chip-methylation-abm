"""
Helper functions for computing methylation heterogeniety metrics computed over a 
2d gene x CpG array of booleans. 

TODO:

- I asked Gemini to convert R functions into numpy versions and still need to check if they're all 
actually correct (if they ever were)
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


# Helper functions for computing methylation metrics on the entire 2d gene x CpG matrix
def mean_methylation(agent):
    """Mean methylation of the 0/1 matrix."""
    return np.mean(agent.cpgs)


def pattern_freq(agent):
    """Compute the k-mer frequencies (sum of 1s per row)."""
    counts = np.sum(agent.cpgs, axis=1)
    return np.bincount(counts, minlength=agent.n_cpgs + 1) / agent.n_genes


def jsd_unmeth(agent):
    """
    Compute JSD using an 'unmeth' reference distribution.
    Note: scipy's jensenshannon returns the distance, so we square it to get the divergence.
    """
    freqs = pattern_freq(agent)
    
    # Create the 'unmeth' reference (1 at index 0, 0s elsewhere)
    ref_vec = np.zeros(agent.n_cpgs + 1)
    ref_vec[0] = 1.0
    
    # JSD = distance^2
    return jensenshannon(freqs, ref_vec) ** 2


def mean_shannon(agent):
    """
    Compute the unweighted mean of Shannon entropy for columns.
    Since Mesa arrays have no NAs, we don't need the weighted mean logic from R.
    """
    p_me = np.mean(agent.cpgs, axis=0)
    p_un = 1.0 - p_me
    
    # Temporarily ignore divide-by-zero warnings for log2(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -1 * ((p_me * np.log2(p_me)) + (p_un * np.log2(p_un)))
        
    # Replace NaNs (caused by 0 * log(0)) with 0.0
    h = np.nan_to_num(h, nan=0.0)
    
    return np.mean(h)


def mean_pdr(agent):
    """Compute the mean proportion of discordant reads (PDR)."""
    p_me = np.mean(agent.cpgs, axis=0)
    p_un = 1.0 - p_me
    pdr_vals = 2 * p_me * p_un

    return np.mean(pdr_vals)


def beta_var(agent):
    """Compute the variance of a vector of beta-values (column means)."""
    betas = np.mean(agent.cpgs, axis=0)
    return np.var(betas, ddof=1) if len(betas) > 1 else 0.0


def gini(agent):
    """Compute the Gini mean difference for read-level beta-values."""
    read_betas = np.mean(agent.cpgs, axis=1)
    
    # Gini Mean Difference formula: mean of all absolute pairwise differences
    # np.subtract.outer creates a matrix of all pairwise differences instantly
    diffs = np.abs(np.subtract.outer(read_betas, read_betas))

    return np.mean(diffs)


def alpha_diversity(agent):
    """
    Compute alpha diversity (Shannon) on the unique read counts.
    """
    # Find unique rows (species) and get their counts
    _, counts = np.unique(agent.cpgs, axis=0, return_counts=True)
  
    return entropy(counts, base=np.e)