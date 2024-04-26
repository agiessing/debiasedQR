\name{sisrq}
\alias{sisrq}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Iterative sure indendence screening for linear quantile regression
}
\description{
  Iterative sure indendence screening for linear quantile regression to improve and robustify the estiamtion of the density matrix.
}
\usage{
sisrq(Y, X, tau, kM, max_iter = 100)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{Responses.
}
  \item{X}{Covariates.
}
  \item{tau}{Quantile level "\eqn{\tau \in (0,1)}".
}
  \item{kM}{Upper bound on (desired) sparsity.
}
  \item{max_iter}{
Maximum number of iterations of sure independence screening. Default value is 100.
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

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
\keyword{Utilities}
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
\concept{Iterative sure independence screening}
% \concept{ ~cpt2 }
% Use only one concept per line.
