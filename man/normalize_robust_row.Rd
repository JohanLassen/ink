% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functions.R
\name{normalize_robust_row}
\alias{normalize_robust_row}
\title{Robust row normalization}
\usage{
normalize_robust_row(ms)
}
\arguments{
\item{ms}{List object containing value data and rowinfo data (meta data)}
}
\value{
ms
}
\description{
Only stable features used to normalize the rows (each sample)
Corrects matrix effects
}
\examples{
# First convert the pneumonia object to a tibble.
library(dplyr)
pneumonia<- tibble::tibble(pneumonia)
# Generate list object
ms <- list()
# Assign feature values to ms$values
start_column <- 8 # The first column with feature values
end_column <- ncol(pneumonia) # The last column of the dataset
ms$values <- pneumonia[1:100, start_column:end_column]
# Assign metadata to ms$rowinfo
ms$rowinfo <- pneumonia[1:100,] \%>\% dplyr::select(id, group, age, gender)
ms <- impute_zero(ms)
ms <- normalize_robust_row(ms)
}
