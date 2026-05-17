import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_simulation_metrics(df: pd.DataFrame, output_path: str = "run_results.png"):
    """
    Generates a basic 3x3 grid plot of simulation metrics across all runs.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if "Overall_Count" not in df.columns:
        df["Overall_Count"] = df["Cell_Count"] + df["Clone_Count"]
    
    metrics_config = [
        {
            "ylabel": "Number of Agents",
            "vars": ["Overall_Count", "Cell_Count", "Clone_Count"],
            "strip": "_Count"
        },
        {
            "ylabel": "Mean Methylation",
            "vars": ["Overall_Mean_Methylation", "Cell_Mean_Methylation", "Clone_Mean_Methylation"],
            "strip": "_Mean_Methylation"
        },
        {
            "ylabel": "JSD",
            "vars": ["Overall_Mean_JSD", "Cell_Mean_JSD", "Clone_Mean_JSD"],
            "strip": "_Mean_JSD"
        }
    ]
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9), sharex=True, sharey="row")
    categories = ["Overall", "Cell", "Clone"]
    
    for row_idx, config in enumerate(metrics_config):
        melt_vars = ["RunId", "Step"] + config["vars"]
        melted_df = df[melt_vars].melt(
            id_vars=["RunId", "Step"],
            var_name="variable",
            value_name="value"
        )
        
        melted_df["variable"] = melted_df["variable"].str.replace(config["strip"], "", regex=False)
        
        for col_idx, cat in enumerate(categories):
            ax = axes[row_idx, col_idx]
            subset = melted_df[melted_df["variable"] == cat]
            
            # 2. Plot the individual stochastic runs (Background)
            sns.lineplot(
                data=subset,
                x="Step",
                y="value",
                units="RunId",
                estimator=None,  
                alpha=0.15,
                color="slategray", # Softer than black
                linewidth=0.8,
                ax=ax,
                zorder=1 # Push to background
            )
            
            # 3. Overlay the Mean trajectory (Foreground)
            sns.lineplot(
                data=subset,
                x="Step",
                y="value",
                estimator="mean", 
                errorbar=None,     # Turn off confidence intervals to avoid clutter
                color="firebrick", # High contrast against the slategray
                linewidth=2.5,
                ax=ax,
                zorder=2 # Pull to front
            )
            
            # 4. Clean up axes (Tufte aesthetics)
            sns.despine(ax=ax)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5) # Horizontal gridlines only
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Basic Titles and Labels
            if row_idx == 0:
                ax.set_title(cat, fontweight="bold", pad=15, fontsize=16)
            
            if col_idx == 0:
                ax.set_ylabel(config["ylabel"], fontsize=14)
            else:
                ax.set_ylabel("")
                
            if row_idx == 2:
                ax.set_xlabel("Step (Years)", fontsize=14)
            else:
                ax.set_xlabel("")
                
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)