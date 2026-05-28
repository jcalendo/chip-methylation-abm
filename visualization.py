import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_simulation_metrics(df: pd.DataFrame, output_path: str = "run_results.png"):
    """
    Generates a 3-panel vertical stacked plot of simulation metrics.
    Overall, Cell, and Clone dynamics are overlaid using distinct colors.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if "Overall_Count" not in df.columns:
        df["Overall_Count"] = df["Cell_Count"] + df["Clone_Count"]
    
    metrics_config = [
        {
            "ylabel": "Number of Agents",
            "vars": ["Overall_Count", "Cell_Count", "Clone_Count"],
            "strip": "_Count",
            "title": "Agent Count"
        },
        {
            "ylabel": "Mean Methylation",
            "vars": ["Overall_Mean_Methylation", "Cell_Mean_Methylation", "Clone_Mean_Methylation"],
            "strip": "_Mean_Methylation",
            "title": "Mean Methylation"
        },
        {
            "ylabel": "JSD",
            "vars": ["Overall_Mean_JSD", "Cell_Mean_JSD", "Clone_Mean_JSD"],
            "strip": "_Mean_JSD",
            "title": "JSD"
        }
    ]
    
    # Define consistent colors for the categories
    palette = {
        "Overall": "dimgray", 
        "Cell": "steelblue", 
        "Clone": "firebrick"
    }
    
    # Create a 3x1 vertical stack with a shared X-axis
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6), sharex=True)
    
    for row_idx, config in enumerate(metrics_config):
        ax = axes[row_idx]
        
        melt_vars = ["RunId", "Step"] + config["vars"]
        melted_df = df[melt_vars].melt(
            id_vars=["RunId", "Step"],
            var_name="Category",
            value_name="value"
        )
        
        melted_df["Category"] = melted_df["Category"].str.replace(config["strip"], "", regex=False)
        
        # Drop NAs to prevent Seaborn plotting errors on empty clone runs
        subset_clean = melted_df.dropna(subset=["value"])
        
        if not subset_clean.empty:
            # Plot the individual stochastic runs (Background)
            sns.lineplot(
                data=subset_clean,
                x="Step",
                y="value",
                hue="Category",
                units="RunId",
                estimator=None,  
                alpha=0.15,
                linewidth=0.8,
                palette=palette,
                legend=False, # Suppress legend for the background spaghetti
                ax=ax,
                zorder=1 
            )
            
            # Overlay the Mean trajectory (Foreground)
            sns.lineplot(
                data=subset_clean,
                x="Step",
                y="value",
                hue="Category",
                estimator="mean", 
                errorbar=None,     
                linewidth=3.0,     
                palette=palette,
                legend=(row_idx == 1), # Draw legend on the MIDDLE plot (index 1)
                ax=ax,
                zorder=2 
            )
        
        # Clean up axes
        sns.despine(ax=ax)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5) 
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Titles and Labels
        ax.set_title(config["title"], fontweight="bold", pad=15, fontsize=16)
        ax.set_ylabel(config["ylabel"], fontsize=14)
        
        # Apply the X-axis label to ALL plots since they are side-by-side
        ax.set_xlabel("Step (Years)", fontsize=14) 
            
    # Format the single-row legend on the middle plot
    if axes[1].get_legend() is not None:
        sns.move_legend(
            axes[1], "upper center", 
            bbox_to_anchor=(0.5, -0.15), # Drop it below the x-axis
            ncol=3,                      # Force it into a single row of 3 items
            title=None,                  # Remove title for a cleaner horizontal look
            frameon=False,
            fontsize=12
        )
            
    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)