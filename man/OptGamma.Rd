\name{optGamma}
\alias{optGamma}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Optimal tuning parameter
}
\description{
  Finds the optimal tuning parameter for primal and dual program from a list of
cross-validated losses.
}
\usage{
optGamma(dual_loss, gamma_lst, cv_fold, cv_rule, robust = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dual_loss}{
  List of values of the cross-validated dual objective function.
}
  \item{gamma_lst}{
  List of tuning parameters "\eqn{\gamma > 0}" for the primal debiasing program from which to choose the optimal value. The default value NULL results in a list with 41 equally spaced values from 0 to the maximum norm of the querry point x. If manually set, end point should always be the maximum norm of the querry point x.
}
  \item{cv_fold}{
  Number of folds of the cross-validation procedure.
}
  \item{cv_rule}{Cross-validation rule to be applied to .drqcv object to select optimal tuning parameter "\eqn{gamma >0}". Choices are `1se', `mincv', and `minfeas'. For details see documentatio of optGamma().
}
  \item{robust}{
  If robust = TRUE, then median and mean absolute deviation (instead of mean and standard deviation) are used when finding the optimal tuning parameter.
}
}

\value{
Optimal tuning parameter "\eqn{\gamma > 0}" of the primal problem chosen via cross-valdiation or simply the provided value.
}
\references{
  Giessing, A. and Wang, J. (2023). Debiased inference on heterogeneous quantile treatment effects with regression rank-scores. \emph{Journal of the Royal Statistical Society: Series B (Statistical Methodology)}, 85(5): 1561–1588. \url{https://doi.org/10.1093/jrsssb/qkad075}
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
\concept{Optimal cross-validated tuning parameter}
% \concept{ ~cpt2 }
% Use only one concept per line.
