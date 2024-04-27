#' Tuning parameter Selection for l1-penalized QR
#'
#' Finds a regularization parameter for L1-penalized linear quantile regression
#' by exploiting that the sup-norm of the gradient of the check loss is a pivot.
#'
#' Belloni, A. and Chernozhukov, V. (2011). L1-penalized quantile regression in
#' high-dimensional sparse models. #' Annals of Statistics. 39 (1) 82 - 130.
#' \url{https://doi.org/10.1214/10-AOS827}

#' @import stats
#' @export

lambdaBC <- function(X, R = 1000, tau = tau, c = 1, alpha = .1, intercept = FALSE){
  norm2n <- function(z){  sqrt(mean(z^2)) }
  n <- nrow(X)
  sigs  <- apply(X, 2, norm2n)
  U <- matrix(runif(n * R),n)
  R <- (t(X) %*% (tau - (U < tau)))/(sigs*sqrt(tau*(1-tau)))
  r <-  apply(abs(R),2,max, na.rm = TRUE)
  c * quantile(r, 1 - alpha) * sqrt(tau*(1-tau))*c(as.numeric(intercept),sigs)
}
