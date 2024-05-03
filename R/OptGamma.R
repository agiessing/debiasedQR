#' This function finds the optimal tuning paramter gamma from a list of cross-
#' valdiated dual losses.
#'
#' @param dual_loss   List of values of the cross-validated dual objective function.
#' @param gamma_lst   List of tuning parameters "\eqn{\gamma > 0}" for the primal
#'                    debiasing program from which to choose the optimal value.
#' @param cv_fold     Cross-validation rule to be applied to .drqcv object to
#'                    select optimal tuning parameter "\eqn{gamma >0}".
#'                    Default value is 5.
#' @param cv_rule     Cross-validation rule to be applied to .drqcv object to
#'                    select optimal tuning parameter "\eqn{gamma >0}". Choices
#'                    are `1se', `mincv', and `minfeas'.
#' @param robust      Only relevant for .drqcv objects. If robust = TRUE, then the
#'                    largest and smallest dual losses are ignored when computing
#'                    the average and standard deviation of the dual losses. If
#'                    robust = FALSE, then all dual losses are used.
#' @return gamma_opt  Optimal tuning parameter "\eqn{\gamma > 0}" of the primal
#'                    problem chosen via cross-validation

#' @export

optGamma <- function(dual_loss, gamma_lst, cv_fold, cv_rule, robust = TRUE) {

  if (is.null(gamma_lst)) {
    stop(paste0("Missing list of tuning parameters!\n"))
  }

  robustMean <- function(x) {
    min <- which.min(x)
    max <- which.max(x)
    mean(x[-c(min, max)], na.rm=FALSE)
  }

  robustSd <- function(x) {
    min <- which.min(x)
    max <- which.max(x)
    sd(x[-c(min, max)], na.rm=FALSE)
  }

  if (robust == TRUE) {
    mean_dual_loss <- apply(dual_loss, 2, FUN = robustMean)
    se_dual_loss <- apply(dual_loss, 2, FUN = robustSd) / sqrt(cv_fold-2)
  } else {
    mean_dual_loss <- apply(dual_loss, 2, mean, na.rm = FALSE)
    se_dual_loss <- apply(dual_loss, 2, function(x){sd(x, na.rm = FALSE)}) / sqrt(cv_fold)
  }

  if (cv_rule == "mincv") {
    gamma_opt <- gamma_lst[which.min(mean_dual_loss)]
  }
  if (cv_rule == "1se") {
    One_SE <- (mean_dual_loss > min(mean_dual_loss, na.rm = TRUE) + se_dual_loss[which.min(mean_dual_loss)]) &
      (gamma_lst < gamma_lst[which.min(mean_dual_loss)])
    if (sum(One_SE, na.rm = TRUE) == 0) {
      One_SE <- rep(TRUE, length(gamma_lst))
    }
    gamma_lst <- gamma_lst[One_SE]
    gamma_opt <- gamma_lst[which.min(mean_dual_loss[One_SE])]
  }
  if (cv_rule == "minfeas") {
    gamma_opt <- min(gamma_lst[!is.na(mean_dual_loss)])
  }

  return(gamma_opt)
}
