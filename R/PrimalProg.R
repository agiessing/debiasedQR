#' @import MASS
#' @rawNamespace import(CVXR, except = c(power, huber))
#' @export

primal <- function(X, x, Psi, gamma = 0.1, quiet = TRUE) {
  n <- dim(X)[1]
  d <- dim(X)[2]
  x <- array(x, dim = c(1,d))
  if (length(gamma) == 0) {
    gamma <- 1
  }

  mask <- rep(TRUE, n)
  mask[which(diag(Psi) == Inf)] <- FALSE
  w <- Variable(rows = sum(mask), cols = 1)
  primal_obj <- Minimize(quad_form(w, diag(diag(Psi)[mask]) ))
  constraints <- list(x - (1/sqrt(n))*(t(w) %*%  X[mask,]) <= gamma,
                     x - (1/sqrt(n))*(t(w) %*% X[mask,]) >= -gamma)
  primal_prog <- Problem(primal_obj, constraints)

  tryCatch({
      res <- psolve(primal_prog)
    }, error = function(e) {
      return(matrix(NA, nrow = n, ncol = 1))
  })

  tryCatch({
    if(res$value == Inf) {
      if(!quiet){
        message("Primal debiasing program is infeasible! Returning 'NA'...\n")
      }
      return(matrix(NA, nrow = n, ncol = 1))
    } else if (sum(res[[1]] == "solver_error") > 0){
      warning("'CVXR' cannot solve this program! Returning 'NA'...")
      return(matrix(NA, nrow = n, ncol = 1))
    }
    else {
      w_large <- rep(0, n)
      w_large[mask] <- res$getValue(w)
      return(w_large)
    }
  }, error = function(e){
    warning("'CVXR' cannot solve this program! Returning 'NA'...")
    return(matrix(NA, nrow = n, ncol = 1))
  })
}
