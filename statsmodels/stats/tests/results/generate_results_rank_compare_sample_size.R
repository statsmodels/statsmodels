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

generate_result_matrices <- function(x1, x2, input_alpha, power, t) {
  result_matrices <- lapply(
    # Since WMWSSP only supports the two-sided case, multiple input alpha by 2 for the one-sided test
    X = c(input_alpha, input_alpha * 2),
    FUN = function(alpha) {
      result_matrix <- WMWSSP(x1 = x1, x2 = x2, alpha = alpha, power = power, t = t)
      # This is scoped from the enclosing environment of the function
      rownames(result_matrix) <- matrix_row_names
      # Reset alpha back to the input alpha level, so it can be used for testing
      result_matrix["alpha", ] <- input_alpha
      return(result_matrix)
    }
  ) |>
    `names<-`(c("two-sided", "one-sided"))
  return(result_matrices)
}

# Seizure example ---------------------------------------------------------

seizure_reference <- c(3, 3, 5, 4, 21, 7, 2, 12, 5, 0, 22, 4, 2, 12, 9, 5, 3, 29, 5, 7, 4, 4, 5, 8, 25, 1, 2, 12)
seizure_synthetic <- c(1, 1, 2, 2, 10, 3, 1, 6, 2, 0, 11, 2, 1, 6, 4, 2, 1, 14, 2, 3, 2, 2, 2, 4, 12, 0, 1, 6)
seizure_alpha <- 0.05
seizure_power <- 0.8
seizure_t <- 0.5
seizure_result_matrices <- generate_result_matrices(
  x1 = seizure_reference,
  x2 = seizure_synthetic,
  input_alpha = seizure_alpha,
  power = seizure_power,
  t = seizure_t
)

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
nasal_mucosa_t <- 0.5
nasal_mucosa_result_matrices <- generate_result_matrices(
  x1 = nasal_mucosa_reference,
  x2 = nasal_mucosa_synthetic,
  input_alpha = nasal_mucosa_alpha,
  power = nasal_mucosa_power,
  t = nasal_mucosa_t
)

# Kidney weight example --------------------------------------------------

kidney_weight_placebo <- c(6.62, 6.65, 5.78, 5.63, 6.05, 6.48, 5.50, 5.37)
kidney_weight_drug <- c(6.92, 6.95, 6.08, 5.93, 6.35, 6.78, 5.80, 5.67)
kidney_weight_alpha <- 0.05
kidney_weight_power <- 0.8
kidney_weight_t <- 0.5
kidney_weight_result_matrices <- generate_result_matrices(
  x1 = kidney_weight_placebo,
  x2 = kidney_weight_drug,
  input_alpha = kidney_weight_alpha,
  power = kidney_weight_power,
  t = kidney_weight_t
)

# Generate results --------------------------------------------------------

results <- data.frame(
  "seizure_two_sided" = seizure_result_matrices[["two-sided"]],
  "seizure_one_sided" = seizure_result_matrices[["one-sided"]],
  "nasal_mucosa" = nasal_mucosa_result_matrices[["two-sided"]],
  "nasal_mucosa" = nasal_mucosa_result_matrices[["one-sided"]],
  "kidney_weight" = kidney_weight_result_matrices[["two-sided"]],
  "kidney_weight" = kidney_weight_result_matrices[["one-sided"]]
) |>
  t() |>
  as.data.frame()

results[["alternative"]] <- rep(c("two-sided", "one-sided"), 3)

# Add reference and synthetic samples as comma separated strings
results[["reference_sample"]] <- c(
  paste0(seizure_reference, collapse = ","),
  paste0(seizure_reference, collapse = ","),
  paste0(nasal_mucosa_reference, collapse = ","),
  paste0(nasal_mucosa_reference, collapse = ","),
  paste0(kidney_weight_placebo, collapse = ","),
  paste0(kidney_weight_placebo, collapse = ",")
)
results[["synthetic_sample"]] <- c(
  paste0(seizure_synthetic, collapse = ","),
  paste0(seizure_synthetic, collapse = ","),
  paste0(nasal_mucosa_synthetic, collapse = ","),
  paste0(nasal_mucosa_synthetic, collapse = ","),
  paste0(kidney_weight_drug, collapse = ","),
  paste0(kidney_weight_drug, collapse = ",")
)
rownames(results) <- NULL

# Keep only columns needed for testing
results <- results[c("alpha", "power", "prop_reference", "alternative", "n_total", "nobs1", "nobs2", "reference_sample", "synthetic_sample", "relative_effect")]

# Save results -----------------------------------------------------------

parser <- ArgumentParser(description = "Generate reference implementation results for rank compare sample size")
parser$add_argument("--output_path", help = "Absolute output file path", default = "results_rank_compare_sample_size.csv")
args <- parser$parse_known_args()[[1]]
write.csv(results, file = args$output_path, row.names = FALSE)
