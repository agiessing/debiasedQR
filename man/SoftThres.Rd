\name{SoftThres}
\alias{SoftThres}

\title{
  Soft-thresholding function
}

\description{
  Proximal operator of the L1-norm, needed for the (proximal) coordinate descent and alternating direction of multiplier method algorithms.
}

\usage{
SoftThres(theta, lambda)
}

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
