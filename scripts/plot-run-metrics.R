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
run_dt[, Overall_Count := Cell_Count + Clone_Count]

count_dt <- melt(
  run_dt,
  id.vars = c("RunId", "Step"),
  measure.vars = c("Overall_Count", "Cell_Count", "Clone_Count"),
  variable.factor = FALSE
)
count_dt[, variable := gsub("_Count", "", variable)]

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
  measure.vars = c("Overall_Mean_JSD", "Cell_Mean_JSD", "Clone_Mean_JSD"),
  variable.factor = FALSE
)
jsd_dt[, variable := gsub("_Mean_JSD", "", variable)]

# Methylation variance
var_dt <- melt(
  run_dt,
  id.vars = c("RunId", "Step"),
  measure.vars = c(
    "Overall_Methylation_Var",
    "Cell_Methylation_Var",
    "Clone_Methylation_Var"
  ),
  variable.factor = FALSE
)
var_dt[, variable := gsub("_Methylation_Var", "", variable)]


# Plot the metric values across all runs ---------------------------------------------------------

plot_metric <- function(dt, ...) {
  ggplot(dt, aes(x = Step, y = value)) +
    geom_line(aes(group = RunId), alpha = 0.1) +
    scale_x_continuous(n.breaks = 10) +
    labs(x = "Step", ...) +
    facet_wrap(~variable) +
    theme_classic() +
    theme(
      legend.position = "bottom",
      panel.grid.major = element_line(linetype = 2, color = "grey80")
    )
}

p <- wrap_plots(
  plot_metric(count_dt, title = "Cell Counts", y = "Number of Agents"),
  plot_metric(
    meth_dt,
    title = "Mean Methylation",
    y = "Mean Methylation (beta-value)"
  ),
  plot_metric(jsd_dt, title = "JSD", y = "JSD"),
  plot_metric(var_dt, title = "Methylation Variance", y = "Variance") +
    labs(caption = paste0("Showing data for all:", n_runs, " simulated runs.")),
  nrow = 4
) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(
  filename = here("results", "run_results.png"),
  plot = p,
  width = 12,
  height = 12,
  dpi = 600
)
