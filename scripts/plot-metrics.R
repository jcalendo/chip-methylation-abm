suppressPackageStartupMessages(library(here))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(tinyplot))


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
jsd_dt[is.na(value), value := 0.0]

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
  plt(
    value ~ V1 | variable,
    data = dt,
    xlab = "Step",
    type = "l",
    theme = "clean2",
    legend = legend("bottom!", title = ""),
    ...
  )
}

png(
  here("results", "model-metrics.png"),
  width = 20,
  height = 12,
  units = "in",
  res = 600
)
tpar(mfrow = c(2, 3))
plot_metric(count_dt, main = "Cell Counts", ylab = "Number of Agents")
plot_metric(meth_dt, main = "Mean Methylation", ylab = "Mean Methylation")
plot_metric(var_dt, main = "Methylation Variance", ylab = "Variance")
plot_metric(jsd_dt, main = "Population JSD", ylab = "JSD")
plot_metric(drift_dt, main = "Population JSD (Drift)", ylab = "JSD")
dev.off()
