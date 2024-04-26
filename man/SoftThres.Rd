\name{SoftThres}
\alias{SoftThres}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Soft-thresholding function
}
\description{
  Proximal operator of the L1-norm, needed for the (proximal) coordinate descent and alternating direction of multiplier method algorithms.
}
\usage{
SoftThres(theta, lambda)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{theta}{Vector to be thesholded.
}
  \item{lambda}{Penalty paramter.
}
}

\value{Thresholded value of theta.
}

\author{
  Alexander Giessing, \email{giessing@umich.edu}
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
