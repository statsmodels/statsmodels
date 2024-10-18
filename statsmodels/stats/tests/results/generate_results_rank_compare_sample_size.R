# Run as: Rscript generate_results_samplesize_rank_compare_onetail.R --output_path path/to/results_samplesize_rank_compare_onetail.csv
if (!require(rankFD)) {
  install.packages("rankFD")
  library(rankFD)
} else if (!require(argparse)) {
  install.packages("argparse")
  library(argparse)
} else if (!require(data.table)) {
  install.packages("data.table")
  library(data.table)
} else if (!require(purrr)) {
  install.packages("purrr")
  library(purrr)
}

generate_result_matrices <- function() {
  # Three examples taken from the reference paper
  inputs_grid <- list(
    x1 = list(
      # Seizure example
      c(3, 3, 5, 4, 21, 7, 2, 12, 5, 0, 22, 4, 2, 12, 9, 5, 3, 29, 5, 7, 4, 4, 5, 8, 25, 1, 2, 12),
      # Nasal mucosa example
      c(
        rep.int(x = 0, times = 64),
        rep.int(x = 1, times = 12),
        rep.int(x = 2, times = 4),
        rep.int(x = 3, times = 0)
      ),
      # Kidney weight example
      c(6.62, 6.65, 5.78, 5.63, 6.05, 6.48, 5.50, 5.37)
    ),
    x2 = list(
      # Seizure example
      c(1, 1, 2, 2, 10, 3, 1, 6, 2, 0, 11, 2, 1, 6, 4, 2, 1, 14, 2, 3, 2, 2, 2, 4, 12, 0, 1, 6),
      # Nasal mucosa example
      c(
        rep.int(x = 0, times = 48),
        rep.int(x = 1, times = 25),
        rep.int(x = 2, times = 6),
        rep.int(x = 3, times = 1)
      ),
      # Kidney weight example
      c(6.92, 6.95, 6.08, 5.93, 6.35, 6.78, 5.80, 5.67)
    ),
    input_alpha = c(0.05, 0.05, 0.05),
    power = c(0.8, 0.8, 0.8),
    t = c(0.5, 0.55, 0.45),
    # Seizure example -> relative effect < 0.5
    # Nasal mucosa example -> relative effect > 0.5
    # Kidney weight example -> relative effect > 0.5
    onetail_alternative = c("smaller", "larger", "larger")
  )

  data <- pmap(
    .l = inputs_grid,
    .f = function(x1, x2, input_alpha, power, t, onetail_alternative) {
      one_two_sided_data <- map(
        .x = c("two-sided", "one-sided"),
        .f = function(alternative) {
          # Match the 'alternative' options in the python function (i.e., 'two-sided', 'smaller', or 'larger')
          alternative <- ifelse(alternative == "two-sided", alternative, onetail_alternative)
          # Since WMWSSP only supports the two-sided case, multiply input alpha by 2 for the one-sided case
          alpha <- ifelse(alternative == "two-sided", input_alpha, input_alpha * 2)
          result_matrix <- WMWSSP(x1 = x1, x2 = x2, alpha = alpha, power = power, t = t)
          # Reset alpha back to the input alpha level, so it can be used for testing
          result_matrix["alpha (2-sided)", ] <- input_alpha
          # Unfortunate extra work in transpose as WMWSSP returns a column matrix
          result_row <- result_matrix |>
            t() |>
            as.data.table()
          # Add data samples and alternative as columns to the result row
          result_row[, c("reference_sample", "synthetic_sample", "alternative") := .(paste0(x1, collapse = ","), paste0(x2, collapse = ","), alternative)]
          return(result_row)
        }
      ) |>
        # Rowbind the one-sided and two-sided rows into a data.table with two rows
        rbindlist()
    }
  ) |>
    # Rowbind all data.tables (one per example) into a single data.table
    rbindlist()

  # Rename for consistency with python tests
  setnames(x = data, old = names(data), new = c(
    "alpha",
    "power",
    "relative_effect",
    "nobs_total",
    "prop_reference",
    "nobs_ref",
    "nobs_treat",
    "nobs_total_rounded",
    "nobs_ref_rounded",
    "nobs_treat_rounded",
    "reference_sample",
    "synthetic_sample",
    "alternative"
  ))

  # Keep only columns needed for testing
  data[, c("nobs_total_rounded", "nobs_ref_rounded", "nobs_treat_rounded") := NULL]

  return(data)
}

test_data <- generate_result_matrices()

# Save results -----------------------------------------------------------

parser <- ArgumentParser(description = "Generate reference implementation results for rank compare sample size")
parser$add_argument("--output_path", help = "Absolute output file path", default = "results_samplesize_rank_compare_onetail.csv")
args <- parser$parse_known_args()[[1]]
fwrite(x = test_data, file = args$output_path, row.names = FALSE)
