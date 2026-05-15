import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pathlib import Path


def plot_simulation_metrics(df: pd.DataFrame, output_path: str = "run_results.png"):
    """
    Generates a 4x3 grid plot of simulation metrics across all runs, 
    mimicking the layout of the original ggplot2/patchwork R script.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    n_runs = df["RunId"].max() + 1
    
    if "Overall_Count" not in df.columns:
        df["Overall_Count"] = df["Cell_Count"] + df["Clone_Count"]
    
    metrics_config = [
        {
            "ylabel": "Number of Agents",
            "vars": ["Overall_Count", "Cell_Count", "Clone_Count"],
            "strip": "_Count"
        },
        {
            "ylabel": "Mean Methylation\n(beta-value)",
            "vars": ["Overall_Mean_Methylation", "Cell_Mean_Methylation", "Clone_Mean_Methylation"],
            "strip": "_Mean_Methylation"
        },
        {
            "ylabel": "JSD",
            "vars": ["Overall_Mean_JSD", "Cell_Mean_JSD", "Clone_Mean_JSD"],
            "strip": "_Mean_JSD"
        },
        {
            "ylabel": "Variance",
            "vars": ["Overall_Methylation_Var", "Cell_Methylation_Var", "Clone_Methylation_Var"],
            "strip": "_Methylation_Var"
        }
    ]
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 12), sharex=True, sharey="row")
    categories = ["Overall", "Cell", "Clone"]
    
    for row_idx, config in enumerate(metrics_config):
        melted_df = df.melt(
            id_vars=["RunId", "Step"],
            value_vars=config["vars"],
            var_name="variable",
            value_name="value"
        )
        
        melted_df["variable"] = melted_df["variable"].str.replace(config["strip"], "", regex=False)
        
        for col_idx, cat in enumerate(categories):
            ax = axes[row_idx, col_idx]
            
            subset = melted_df[melted_df["variable"] == cat]
            
            # Plot all runs using seaborn
            sns.lineplot(
                data=subset,
                x="Step",
                y="value",
                units="RunId",
                estimator=None,  # Critical: disables slow bootstrapping, draws individual lines
                alpha=0.1,
                color="black",
                ax=ax
            )
            
            # Apply ggplot theme-like elements
            ax.grid(True, linestyle="--", color="grey", alpha=0.3)
            sns.despine(ax=ax)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10)) # Equivalent to n.breaks=10
            
            # Set Titles and Labels
            if row_idx == 0:
                ax.set_title(cat, fontsize=12, pad=10)
            
            if col_idx == 0:
                ax.set_ylabel(config["ylabel"], fontsize=10)
            else:
                ax.set_ylabel("")
                
            if row_idx == 3:
                ax.set_xlabel("Step", fontsize=10)
            else:
                ax.set_xlabel("")
                
    # Add the caption at the bottom right
    fig.text(
        0.98, 0.02, 
        f"Showing data for all: {n_runs} simulated runs.", 
        ha="right", fontsize=10, color="grey"
    )
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.06) 
    
    # Save the figure
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)