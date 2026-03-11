#!/usr/bin/env Rscript --vanilla
#
# Create plots of metrics computed across the entire model
#
# ------------------------------------------------------------------------------------------------
suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))


# Read in and clean data -------------------------------------------------------------------------

model_dt <- fread(here("results", "model_results.csv.gz"))

# Cell counts at each step
count_dt <- melt(
  model_dt,
  id.vars = "V1",
  measure.vars = c("Cell_Count", "Clone_Count"),
  variable.factor = FALSE
)
count_dt[, variable := fifelse(variable == "Cell_Count", "Cell", "Clone")]

# Mean methylation
meth_dt <- melt(
  model_dt,
  id.vars = "V1",
  measure.vars = c(
    "Overall_Mean_Methylation",
    "Cell_Mean_Methylation",
    "Clone_Mean_Methylation"
  ),
  variable.factor = FALSE
)
meth_dt[is.na(value), value := 0.0]
meth_dt[, variable := gsub("_Mean_Methylation", "", variable)]

# Variance of mean methylation
var_dt <- melt(
  model_dt,
  id.vars = "V1",
  measure.vars = c(
    "Overall_Methylation_Variance",
    "Cell_Methylation_Variance",
    "Clone_Methylation_Variance"
  ),
  variable.factor = FALSE
)
var_dt[, variable := gsub("_Methylation_Variance", "", variable)]
var_dt[is.na(value), value := 0.0]

# JSD of the population
jsd_dt <- melt(
  model_dt,
  id.vars = "V1",
  measure.vars = c(
    "Overall_Population_JSD",
    "Cell_Population_JSD",
    "Clone_Population_JSD"
  ),
  variable.factor = FALSE
)
jsd_dt[, variable := gsub("_Population_JSD", "", variable)]
jsd_dt[is.na(value), value := 0.0]

# JSD to uniform reference
drift_dt <- melt(
  model_dt,
  id.vars = "V1",
  measure.vars = c("Overall_Drift_JSD", "Cell_Drift_JSD", "Clone_Drift_JSD"),
  variable.factor = FALSE
)
drift_dt[, variable := gsub("_Drift_JSD", "", variable)]
drift_dt[is.na(value), value := 0.0]


# Plot --------------------------------------------------------------------------------------------

plot_metric <- function(dt, ...) {
  ggplot(dt, aes(x = V1, y = value, color = variable)) +
    geom_line(linewidth = 1.5) +
    scale_x_continuous(n.breaks = 10) +
    scale_color_manual(
      values = c(
        "Cell" = "orange",
        "Clone" = "cornflowerblue",
        "Overall" = "red2"
      )
    ) +
    labs(x = "Step", color = "Cell Type", ...) +
    coriell::theme_coriell() +
    theme(legend.position = "bottom")
}

p <- wrap_plots(
  plot_metric(count_dt, title = "Cell Counts", y = "Number of Agents") +
    guides(color = "none"),
  plot_metric(
    meth_dt,
    title = "Mean Methylation",
    y = "Mean Methylation (beta-value)"
  ),
  plot_metric(drift_dt, title = "Population JSD (Drift)", y = "JSD"),
  ncol = 3
) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
ggsave(
  filename = here("results", "model_results.png"),
  plot = p,
  width = 18,
  height = 6,
  dpi = 600
)
