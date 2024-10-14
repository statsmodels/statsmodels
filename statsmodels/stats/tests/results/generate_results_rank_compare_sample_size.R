# Run this script to generate reference implementation results for rank compare sample size
if (!require(rankFD)) {
  install.packages("rankFD")
  library(rankFD)
} else if (!require(argparse)) {
  install.packages("argparse")
  library(argparse)
} 

matrix_row_names <- c(
  "alpha",
  "power",
  "relative_effect",
  "n_total",
  "prop_reference",
  "nobs1",
  "nobs2",
  "n_total_rounded",
  "nobs1_rounded",
  "nobs2_rounded"
)

# Seizure example ---------------------------------------------------------

seizure_reference <- c(3, 3, 5, 4, 21, 7, 2, 12, 5, 0, 22, 4, 2, 12, 9, 5, 3, 29, 5, 7, 4, 4, 5, 8, 25, 1, 2, 12)
seizure_synthetic <- c(1, 1, 2, 2, 10, 3, 1, 6, 2, 0, 11, 2, 1, 6, 4, 2, 1, 14, 2, 3, 2, 2, 2, 4, 12, 0, 1, 6)
seizure_alpha <- 0.05
seizure_power <- 0.8
seisure_ratio <- 0.5
seizure_result_matrix <- WMWSSP(x1 = seizure_reference, x2 = seizure_synthetic, alpha = seizure_alpha, power = seizure_power, t = seisure_ratio)
rownames(seizure_result_matrix) <- matrix_row_names

# Nasal mucosa example ---------------------------------------------------

nasal_mucosa_reference <- c(
  rep.int(x = 0, times = 64),
  rep.int(x = 1, times = 12),
  rep.int(x = 2, times = 4),
  rep.int(x = 3, times = 0)
)
nasal_mucosa_synthetic <- c(
  rep.int(x = 0, times = 48),
  rep.int(x = 1, times = 25),
  rep.int(x = 2, times = 6),
  rep.int(x = 3, times = 1)
)
nasal_mucosa_alpha <- 0.05
nasal_mucosa_power <- 0.8
nasal_mucosa_ratio <- 0.5
nasal_mucosa_result_matrix <- WMWSSP(x1 = nasal_mucosa_reference, x2 = nasal_mucosa_synthetic, alpha = nasal_mucosa_alpha, power = nasal_mucosa_power, t = nasal_mucosa_ratio)
rownames(nasal_mucosa_result_matrix) <- matrix_row_names

# Kidney weight example --------------------------------------------------

kidney_weight_placebo <- c(6.62, 6.65, 5.78, 5.63, 6.05, 6.48, 5.50, 5.37)
kidney_weight_drug <- c(6.92, 6.95, 6.08, 5.93, 6.35, 6.78, 5.80, 5.67)
kidney_weight_alpha <- 0.05
kidney_weight_power <- 0.8
kidney_weight_ratio <- 0.5
kidney_weight_result_matrix <- WMWSSP(x1 = kidney_weight_placebo, x2 = kidney_weight_drug, alpha = kidney_weight_alpha, power = kidney_weight_power, t = kidney_weight_ratio)
rownames(kidney_weight_result_matrix) <- matrix_row_names

# Generate results --------------------------------------------------------

results <- data.frame(
  "seizure" = seizure_result_matrix,
  "nasal_mucosa" = nasal_mucosa_result_matrix,
  "kidney_weight" = kidney_weight_result_matrix
) |>
  t() |>
  as.data.frame()
results[["experiment_name"]] <- c("seizure", "nasal_mucosa", "kidney_weight")
# Add reference and synthetic samples as comma separated strings
results[["reference_sample"]] <- c(
  paste0(seizure_reference, collapse = ","),
  paste0(nasal_mucosa_reference, collapse = ","),
  paste0(kidney_weight_placebo, collapse = ",")
)
results[["synthetic_sample"]] <- c(
  paste0(seizure_synthetic, collapse = ","),
  paste0(nasal_mucosa_synthetic, collapse = ","),
  paste0(kidney_weight_drug, collapse = ",")
)
rownames(results) <- NULL

# Save results -----------------------------------------------------------

parser <- ArgumentParser(description = "Generate reference implementation results for rank compare sample size")
parser$add_argument("--output_path", help = "Absolute output file path", default = "results_rank_compare_sample_size.csv")
args <- parser$parse_known_args()[[1]]
write.csv(results, file = args$output_path, row.names = FALSE)
