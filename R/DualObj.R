#' (Generic) Objective Function of the Dual Debiasing Program.
#'
#' @export
dualObj <- function(X, x, Psi, v, gamma = 0.05) {
  n <- nrow(X)
  A <-  t(X) %*% Psi %*% X
  obj <- t(v) %*% A %*% v / (4 * n) + t(x) %*% v + gamma * sum(abs(v))

  return(obj)
}



