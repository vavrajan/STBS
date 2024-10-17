library(ggplot2)
library(dplyr)
# Assuming that the working directory is ".../STBS"
ROOT <- paste0(getwd(), "/data/hein-daily/fits/")
FIG <- paste0(getwd(), "/data/hein-daily/figs/")

eta_fixed <- read.csv(paste0(ROOT, "STBS_ideal_a_all114_K25/params/eta_ideal_a_variability.csv"))
eta_varying <- read.csv(paste0(ROOT, "STBS_ideal_ak_all114_K25/params/eta_ideal_ak_variability.csv"))

colnames(eta_fixed) <- colnames(eta_varying) <- c("Topic", "Variance")

topic_labels <- c("National Security",
                  "Supreme Court",
                  "Coast Guard",
                  "Human Trafficking",
                  "Commemoration and Anniversaries",
                  "Gun Violence",
                  "Middle Class and Small Businesses",
                  "Health Care",
                  "Public Health (Zika)",
                  "Veterans and Health Care",
                  "Drugs and Addiction",
                  "Climate Change",
                  "Natural Resources",
                  "Planned Parenthood and Abortion",
                  "Institutes and Research",
                  "Middle East and Nuclear Weapons",
                  "Immigration and Department of Homeland Security",
                  "Social Security and Taxes",
                  "Rhetorics and Discussion",
                  "Clean Water Act",
                  "Law Enforcement",
                  "Wars and Human Rights",
                  "Education for Children",
                  "Cyber Security",
                  "Export, Import and Business")

etas <- rbind(cbind(eta_fixed, Model = "Fixed ideological position"),
              cbind(eta_varying, Model = "Topic-specific ideological position")) |>
    dplyr::mutate(Topic = factor(Topic, unique(Topic)),
                  Model = factor(Model),
                  Label = factor(rep(topic_labels,2)))
etas[,"Topic"] = factor(paste0(etas$Label, "   ", etas$Topic),
                        levels = paste0(topic_labels, "   ", 0:24),
                        labels = paste0(topic_labels, ifelse(0:24 < 10, "     ", "   "), 0:24))

polarity_plot <- ggplot2::ggplot(etas, ggplot2::aes(Topic, Variance, fill = Model)) +
    ggplot2::geom_bar(position = "dodge", stat = "identity") +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_brewer(palette = "Set2") +
    ggplot2::theme_bw() +
    # ggplot2::scale_x_continuous(sec.axis = ggplot2::sec_axis(~., breaks=1:25, labels=topic_labels)) +
    ggplot2::theme(axis.title.y = ggplot2::element_blank(),
                   # axis.text.x = ggplot2::element_text(angle=70, vjust=0.6),
                   legend.position = c(.99, .01),
                   #legend.reverse = TRUE,
                   legend.direction = "vertical",
                   legend.justification = c("right", "bottom"),
                   legend.box.just = "left",
                   legend.margin = ggplot2::margin(6, 6, 6, 6),
                   legend.title = ggplot2::element_blank()) +
    ggplot2::guides(fill = ggplot2::guide_legend(reverse=TRUE))


ggplot2::ggsave(paste0(FIG, "eta_ideal_variability_a_vs_ak_R.png"), polarity_plot,
                width = 8, height = 4)

# ordered increasingly by fixed ideological position (top), topic-specific ideological position (bottom)
etas <- etas |>
    dplyr::arrange(Model, Variance) |>
    dplyr::mutate(Topic = factor(Topic, unique(Topic)),
                  Model = factor(Model))
polarity_plot_1 <- ggplot2::ggplot(etas, ggplot2::aes(Topic, Variance, fill = Model)) +
    ggplot2::geom_bar(position = "dodge", stat = "identity") +
    ggplot2::scale_fill_brewer(palette = "Set2") +
    ggplot2::theme_bw() +
    ggplot2::theme(legend.position = c(.01, .99),
                   legend.direction = "horizontal",
                   legend.justification = c("left", "top"),
                   legend.box.just = "left",
                   legend.margin = ggplot2::margin(6, 6, 6, 6),
                   legend.title = ggplot2::element_blank())
etas <- etas |>
    dplyr::arrange(dplyr::desc(Model), Variance) |>
    dplyr::mutate(Topic = factor(Topic, unique(Topic)),
                  Model = factor(Model))
polarity_plot_2 <- ggplot2::ggplot(etas, ggplot2::aes(Topic, Variance, fill = Model)) +
    ggplot2::geom_bar(position = "dodge", stat = "identity") +
    ggplot2::scale_fill_brewer(palette = "Set2") +
    ggplot2::theme_bw() +
    ggplot2::theme(legend.position = c(.01, .99),
                   legend.direction = "horizontal",
                   legend.justification = c("left", "top"),
                   legend.box.just = "left",
                   legend.margin = ggplot2::margin(6, 6, 6, 6),
                   legend.title = ggplot2::element_blank())
gridExtra::grid.arrange(polarity_plot_1, polarity_plot_2, nrow = 2)

