#!/usr/bin/env Rscript
#
# Create plots of metrics computed across all runs of a given simulation
#
# ------------------------------------------------------------------------------------------------
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))


run_dt <- fread(here("results", "run_results.csv.gz"))
n_runs <- run_dt[, max(RunId)] + 1

count_dt <- melt(
  run_dt,
  id.vars = c("RunId", "Step"),
  measure.vars = c("Cell_Count", "Clone_Count"),
  variable.factor = FALSE
)
count_dt[, variable := fifelse(variable == "Cell_Count", "Cell", "Clone")]

# Mean methylation
meth_dt <- melt(
  run_dt,
  id.vars = c("RunId", "Step"),
  measure.vars = c(
    "Overall_Mean_Methylation",
    "Cell_Mean_Methylation",
    "Clone_Mean_Methylation"
  ),
  variable.factor = FALSE
)
meth_dt[, variable := gsub("_Mean_Methylation", "", variable)]

# JSD to uniform reference
jsd_dt <- melt(
  run_dt,
  id.vars = c("RunId", "Step"),
  measure.vars = c("Overall_Drift_JSD", "Cell_Drift_JSD", "Clone_Drift_JSD"),
  variable.factor = FALSE
)
jsd_dt[, variable := gsub("_Drift_JSD", "", variable)]

# Plot the metric values across all runs ---------------------------------------------------------

plot_metric <- function(dt, ...) {
  ggplot(dt, aes(x = Step, y = value)) +
    geom_line(aes(group = RunId), alpha = 0.5) +
    scale_x_continuous(n.breaks = 10) +
    labs(x = "Step", ...) +
    facet_wrap(~variable) +
    coriell::theme_coriell() +
    theme(
      legend.position = "bottom",
      panel.grid.major = element_line(color = "lightgrey"),
      panel.grid.minor = element_blank()
    )
}

p <- wrap_plots(
  plot_metric(count_dt, title = "Cell Counts", y = "Number of Agents"),
  plot_metric(
    meth_dt,
    title = "Mean Methylation",
    y = "Mean Methylation (beta-value)"
  ),
  plot_metric(jsd_dt, title = "JSD", y = "JSD") +
    labs(caption = paste0("Showing data for all:", n_runs, " simulated runs.")),
  nrow = 3
) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(
  filename = here("results", "run_results.png"),
  plot = p,
  width = 16,
  height = 12,
  dpi = 600
)
