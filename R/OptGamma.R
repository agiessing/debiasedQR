#' @export

optGamma <- function(dual_loss, gamma_lst, cv_fold, cv_rule, robust = TRUE) {

  if (is.null(gamma_lst)) {
    stop(paste0("Missing list of tuning parameters!\n"))
  }

  if (robust == TRUE) {
    mean_dual_loss <- apply(dual_loss, 2, median, na.rm = FALSE)
    std_dual_loss <- apply(dual_loss, 2, function(x){mad(x, na.rm = FALSE)}) / sqrt(cv_fold)
  } else {
    mean_dual_loss <- apply(dual_loss, 2, mean, na.rm = FALSE)
    std_dual_loss <- apply(dual_loss, 2, function(x){sd(x, na.rm = FALSE)}) / sqrt(cv_fold)
  }

  if (cv_rule == "mincv") {
    gamma_opt <- gamma_lst[which.min(mean_dual_loss)]
  }
  if (cv_rule == "1se") {
    One_SE <- (mean_dual_loss > min(mean_dual_loss, na.rm = TRUE) + std_dual_loss[which.min(mean_dual_loss)]) &
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
