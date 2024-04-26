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
