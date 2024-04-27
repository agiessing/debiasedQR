\name{sisrq}
\alias{sisrq}

\title{
  Iterative sure indendence screening for linear quantile regression
}

\description{
  Iterative sure indendence screening for linear quantile regression to improve and robustify the estimation of the density matrix.
}

\usage{
sisrq(Y, X, tau, kM, max_iter = 100, quiet = TRUE)
}

\arguments{
  \item{Y}{Responses.
}
  \item{X}{Covariates.
}
  \item{tau}{Quantile level "\eqn{\tau \in (0,1)}".
}
  \item{kM}{Upper bound on (desired) sparsity.
}
  \item{max_iter}{Maximum number of iterations of sure independence screening. Default value is 100.
}
\item{quiet}{If quiet = TRUE, then no messages are displayed.
}
}

\value{
Returns a model of size at most kM (sparsit level).
}

\references{
  Fan, J. and Lv, J. (2008). Sure independence screening for ultrahigh dimensional feature space. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 70: 849-911. \url{https://doi.org/10.1111/j.1467-9868.2008.00674.x}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\keyword{Utilities}
\concept{Iterative sure independence screening}
