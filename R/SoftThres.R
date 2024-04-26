#' @export

SoftThres <- function(theta, lambda) {
  if (is.vector(theta)) {
    if (length(theta) > 1) {
      res <- sign(theta) * pmax(abs(theta) - lambda, 0)
    } else {
      res <- sign(theta) * max(abs(theta) - lambda, 0)
    }
  } else {
    res <- matrix(0, nrow = nrow(theta), ncol = 2)
    res[, 1] <- abs(theta) - lambda
    res <- sign(theta) * apply(res, 1, max)
  }

  return(res)
}
