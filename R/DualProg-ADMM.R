#' @export

dualADMM <- function(X, x, Psi=NULL, v_init=NULL, gamma = 0.2, rho=5, max_iter = 500, tol = c(1e-6, 1e-6), quiet = TRUE) {
  n <- dim(X)[1]
  d <- dim(X)[2]

  if (is.null(Psi)) {
    Psi <- diag(n)
  }

  if (length(gamma) == 0) {
    gamma <- 1
  }

  A <- solve( t(X) %*% Psi %*% X / (2*n) + rho*diag(1, d) )

  if (is.null(v_init)) {
    v_init <- rep(1, d)
  } else {
    v_init <- array(v_init, dim = c(d,1))
  }

  z_new <- v_init
  u_new <- v_init #* rho

  res_primal <- 1
  res_dual <- 1
  eps_primal <- 0
  eps_dual <- 0
  cnt <- 0

  while( (cnt <= max_iter) && ( (res_primal >= eps_primal) || (res_dual >= eps_dual) ) ) {
    z_old <- z_new
    u_old <- u_new #/ rho
    cnt <- cnt + 1

    # ADMM updating rule
    v_new <- - A %*% ( x + rho * (u_old - z_old) )
    z_new <- SoftThres(v_new + u_old, gamma/rho)
    u_new <- u_old + v_new - z_new

    # ADMM stopping criteria
    res_primal <- norm(v_new - z_new, type = "2")
    res_dual <- norm(- rho * (z_new - z_old), type = "2")
    eps_primal <- sqrt(d) * tol[1] + tol[2] * max( norm(v_new, type = "2"), norm(z_new, type ="2") )
    eps_dual <- sqrt(d) * tol[1] + tol[2] * rho * norm(u_new, type ="2")

    if(!quiet){
      message(paste0("ADMM step count: ", cnt, ". Primal residual criterion: ",
                    res_primal - eps_primal, ". Dual residual criterion: ", res_dual - eps_dual, "\n"))
    }

    if ( (cnt > max_iter) ) {
      message(paste0("ADMM algorithm has reached max. no. of iterations for gamma = ", round(gamma, 4),"."))
      }
    }

  return(list(v = v_new))
}
