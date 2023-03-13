#' Log transform data
#' log(x+1)
#' @param ms List object containing value data and rowinfo data (meta data)
#'
#' @return ms
#' @export
#' @importFrom magrittr %>%
#' @import dplyr
#' @examples
#' # First convert the pneumonia object to a tibble.
#' pneumonia<- tibble::tibble(pneumonia)
#'
#' # Generate list object
#' ms <- list()
#'
#' # Assign feature values to ms$values
#' start_column <- 8 # The first column with feature values
#' end_column <- ncol(pneumonia) # The last column of the dataset
#' ms$values <- pneumonia[1:10, start_column:end_column]
#'
#' # Assign metadata to ms$rowinfo
#' ms$rowinfo <- pneumonia[1:10,] %>% dplyr::select(id, group, age, gender)
#' ms <- transform_log(ms)
#' head(ms$values)
#' head(ms$rowinfo)
transform_log <- function(ms){
  ms$values     <- log(ms$values+1) %>% tibble::as_tibble()
  return(ms)
}



#' Robust row normalization
#'
#' Only stable features used to normalize the rows (each sample)
#' Corrects matrix effects
#'
#' @param ms List object containing value data and rowinfo data (meta data)
#'
#' @return ms
#' @export
#' @importFrom magrittr %>%
#' @import dplyr
#' @import tidyr
#' @examples
#' # First convert the pneumonia object to a tibble.
#' library(dplyr)
#' pneumonia<- tibble::tibble(pneumonia)
#' # Generate list object
#' ms <- list()
#' # Assign feature values to ms$values
#' start_column <- 8 # The first column with feature values
#' end_column <- ncol(pneumonia) # The last column of the dataset
#' ms$values <- pneumonia[1:100, start_column:end_column]
#' # Assign metadata to ms$rowinfo
#' ms$rowinfo <- pneumonia[1:100,] %>% dplyr::select(id, group, age, gender)
#' ms <- impute_zero(ms)
#' ms <- normalize_robust_row(ms)
normalize_robust_row <- function(ms) {

  target_info   <- ms$rowinfo
  target_values <- ms$values %>% tibble::as_tibble()

  tmp1 <- target_info
  tmp2 <- target_values[tmp1$rowid,]

  stable_features <-
    tmp1 %>%
    dplyr::bind_cols(tmp2) %>%
    tidyr::pivot_longer(dplyr::starts_with("M")) %>%
    dplyr::group_by(rowid) %>%
    dplyr::mutate(rank=rank(value)) %>%
    dplyr::ungroup() %>%
    dplyr::group_by(name) %>%
    dplyr::summarise(median = median(rank),
                     range = max(rank)-min(rank)) %>%
    dplyr::ungroup() %>%
    dplyr::slice_min(order_by = median, prop = 0.8) %>%
    dplyr::slice_max(order_by = median, prop = 0.8) %>%
    dplyr::slice_min(order_by = range, prop = 0.8)

  raw    <- target_values
  data.x <- raw
  tmp    <- rowSums(target_values %>% dplyr::select(dplyr::any_of(stable_features$name)))
  raw    <- max(raw)*raw / tmp

  ms$values  <- as_tibble(raw)
  ms$rowinfo <- target_info

  return(ms)
}


#' Plot principal components coloured by a variable in data with triplicates
#'
#' @param ms the modeling data (values and metadata)
#' @param color_label the label used for colouring the data points
#' @param palette fx Spectral, Set1, Set2, Set3, Set4, 1, 2, 3, 4 etc.
#' @param tech_rep the column name of the technical replicates
#'
#' @return a pca plot using the first 6 components
#' @export
plot_pca <- function(ms, color_label="rowid", palette = "Spectral", tech_rep = "rowid") {

  tmp1 <- ms$rowinfo
  tmp2 <- ms$values
  if (!"rowid"%in%colnames(tmp1)){tmp1 <- tmp1 |> tibble::rowid_to_column()}
  r    <- prcomp(x = tmp2, retx = TRUE, center = T, scale = T, rank. = 6)

  variance_explained <- summary(r)$importance[2,1:6]
  variance_explained <- round(variance_explained, 3)*100

  pd <- r$x |>
    tibble::as_tibble() |>
    dplyr::bind_cols(tmp1 |> dplyr::select(dplyr::all_of(color_label)))

  plotlist <- list()
  titles <- list()

  for(i in 1:(ncol(r$x)/2)) {

    xvar <- names(pd)[2*i-1]
    yvar <- names(pd)[2*i]
    p1 <-
      ggplot2::ggplot(pd, ggplot2::aes(x=.data[[xvar]], y=.data[[yvar]], fill=.data[[color_label]], group=.data[[tech_rep]]))+
      ggplot2::geom_polygon(aes(fill=.data[[color_label]]), show.legend = F, alpha = 0.6)+
      ggplot2::geom_point(shape=21, color="black", size=2, show.legend = F) +
      ggplot2::labs(fill = fill,
                    x = paste0(xvar, " (", variance_explained[xvar], " %)"),
                    y = paste0(yvar, " (", variance_explained[yvar], " %)"))+
      ggplot2::theme_minimal() +
      ggplot2::theme(axis.title.x = element_text(size = 8),
                     axis.title.y = element_text(size = 8),
                     axis.text.x = element_text(size = 8),
                     axis.text.y = element_text(size = 8))

    if (!is.numeric(ms$rowinfo[[color_label]])){
      p1 <- p1+scale_fill_brewer(palette = palette)+scale_color_brewer(palette = palette)
    } else{
      p1 <- p1+scale_fill_distiller(palette = palette)+scale_color_distiller(palette = palette)
    }
    plotlist[[length(plotlist)+1]] <- p1

    if (i == 1){
      p1 <-
        ggplot2::ggplot(pd, ggplot2::aes_string(x=xvar, y=yvar, fill=color_label))+
        ggplot2::geom_point(shape=21, color="#FFFFFFFF", size=3)+
        ggplot2::theme(legend.text = element_text(size=8), legend.title = element_text(size=8), legend.key=element_blank()) +
        ggplot2::guides(fill = ggplot2::guide_legend(override.aes = list(size = 2)))

      if (!is.numeric(ms$rowinfo[[color_label]])){
        p1 <- p1+scale_fill_brewer(palette = palette)
      } else{
        p1 <- p1+scale_fill_distiller(palette = palette)
      }
      legend <- cowplot::get_legend(p1 + ggplot2::theme(legend.box.margin = ggplot2::margin(0, 0, 0, 0)))
    }
  }

  p1    <- cowplot::plot_grid(plotlist = plotlist, nrow=1)
  p1    <- cowplot::plot_grid(p1, legend, rel_widths = c(3, .4))
  title <- cowplot::ggdraw() + cowplot::draw_label(paste(ncol(tmp2), "Features"), size = 8)
  final <- cowplot::plot_grid(title, p1, nrow=2, rel_heights = c(1, 10))

  return(final)
}

get_regression_preds <- function(x) {
  x[["predictions"]] |>
    bind_rows() |>
    select(pred = lambda.1se, obs)
}
