#!/usr/bin/env Rscript --vanilla
#
# Create a quick plot of the simulation data
#
# -------------------------------------------------------------------------------------------------
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(tinyplot))
suppressPackageStartupMessages(library(here))


result_files <- list.files(
  path = here("results"),
  pattern = "*.csv",
  full.names = TRUE
)

# Pick the most recent result -- I'll change this into a command line arg later
result <- rev(result_files)[1]
dt <- fread(result)

metrics <- c(
  "Mean_Methylation",
  "JSD_Unmeth",
  "Mean_Shannon",
  "Mean_PDR",
  "Beta_Variance",
  "Gini_Mean_Diff",
  "Alpha_Diversity"
)

png(
  here("results", gsub(".csv.gz", ".png", basename(result))),
  width = 12,
  height = 10,
  units = "in",
  res = 300
)
par(mfrow = c(3, 3))
for (m in metrics) {
  plt(
    x = dt[["Step"]],
    y = dt[[m]],
    by = dt[["AgentID"]],
    type = "l",
    legend = "none",
    theme = "clean2",
    xlab = 'Step',
    ylab = m,
    main = gsub("_", " ", m)
  )
}
dev.off()
