% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functions.R
\name{plot_pca}
\alias{plot_pca}
\title{Plot principal components coloured by a variable in data with triplicates}
\usage{
plot_pca(ms, color_label = "rowid", palette = "Spectral", tech_rep = "rowid")
}
\arguments{
\item{ms}{the modeling data (values and metadata)}

\item{color_label}{the label used for colouring the data points}

\item{palette}{fx Spectral, Set1, Set2, Set3, Set4, 1, 2, 3, 4 etc.}

\item{tech_rep}{the column name of the technical replicates}
}
\value{
a pca plot using the first 6 components
}
\description{
Plot principal components coloured by a variable in data with triplicates
}
