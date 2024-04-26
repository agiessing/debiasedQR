\name{lambdaBC}
\alias{lambdaBC}
%- Also NEED an '\alias' for EACH other topic documented here.
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
%- maybe also 'usage' for other objects documented here.
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

\value{Returns a list of penalty parameters of length dim(X) + 1. If intercept = FAlSE, the first entry is 0.
}
\references{
  Belloni, A. and Chernozhukov, V. (2011). L1-penalized quantile regression in high-dimensional sparse models. \emph{Annals of Statistics}. 39 (1) 82 - 130. \url{https://doi.org/10.1214/10-AOS827}
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (x)
{
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
\keyword{Utilities}
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
