\name{lambdaBC}
\alias{lambdaBC}

\title{
  Adaptive choice of penality paramter
}

\description{
Finds the regularization parameter for L1-penalized linear quantile regression
by exploiting that the sup-norm of the gradient of the check loss is a pivot.
}

\usage{
lambdaBC(X, R = 1000, tau = tau, c = 1, alpha = .1, intercept = FALSE)
}

\arguments{
  \item{X}{Covariates.
}
  \item{R}{Number of unifrom random variables to be generated to approxiamte the distribution of the pivot.
}
  \item{tau}{Quantile level "\eqn{\tau \in (0,1)}".
}
  \item{c}{Constant, typically between 1 and 2.
}
  \item{alpha}{Upper quantile level of the privot.
}
  \item{intercept}{
  If intercept = FALSE, then the intercept of the quantile regression program is not penalized. If intercept = TRUE, then the intercept is penalized. Defaul value is intercept = FALSE.
}
}

\value{
Returns a list of penalty parameters of length dim(X) + 1. If intercept = FAlSE, the first entry is 0.
}

\references{
  Belloni, A. and Chernozhukov, V. (2011). L1-penalized quantile regression in high-dimensional sparse models. \emph{Annals of Statistics}. 39 (1) 82 - 130. \url{https://doi.org/10.1214/10-AOS827}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\keyword{Utilities}
