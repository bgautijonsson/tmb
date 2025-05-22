library(tidyverse)
library(bayesplot)
library(posterior)
library(patchwork)
library(scales)
theme_set(
  bggjphd::theme_bggj()
)

plot_dat_iid <- read_csv(here::here("tables", "gevt", "plot_dat_iid.csv"))
plot_dat_copula <- read_csv(here::here("tables", "gevt", "plot_dat_copula.csv"))

plot_dat <- bind_rows(
  plot_dat_iid |> mutate(model = "IID"),
  plot_dat_copula |> mutate(model = "Copula")
) |>
  filter(str_detect(type, "Smooth")) |>
  mutate(
    model = fct_relevel(
      model,
      "IID",
      "Copula"
    )
  )

uk <- bggjphd::get_uk_spatial(scale = "large")

d <- bggjphd::stations |>
  bggjphd::stations_to_sf() |>
  bggjphd::points_to_grid() |>
  inner_join(
    plot_dat,
    by = join_by(proj_x, proj_y)
  ) |>
  select(-contains("station"))


make_plot <- function(var) {
  p <- d |>
    filter(
      variable == var
    ) |>
    mutate(
      model2 = model,
      min_value = min(value),
      max_value = max(value)
    ) |>
    group_by(model2) |>
    group_map(
      \(x, ...) {
        x |>
          ggplot() +
          geom_sf(
            data = uk |> filter(name == "Ireland")
          ) +
          geom_sf(
            aes(fill = value, col = value),
            linewidth = 0.01,
            alpha = 0.6
          ) +
          scale_fill_distiller(
            palette = "RdBu",
            limits = c(unique(x$min_value), unique(x$max_value))
          ) +
          scale_colour_distiller(
            palette = "RdBu",
            limits = c(unique(x$min_value), unique(x$max_value))
          ) +
          labs(
            subtitle = unique(x$model)
          )
      }
    ) |>
    wrap_plots(guides = "collect") +
    plot_annotation(
      title = "Spatial distribution of GEV parameters",
      subtitle = latex2exp::TeX(
        str_c(
          "$",
          var,
          "$"
        )
      )
    )
  ggsave(
    plot = p,
    filename = str_c("Figures/gevt/compare/", var, ".png"),
    width = 8,
    height = 0.6 * 8,
    scale = 1.5
  )
}

c("psi", "mu", "tau", "sigma", "phi", "xi", "gamma", "Delta") |>
  map(make_plot)


p <- plot_dat |>
  filter(
    # str_detect(type, "Max")
  ) |>
  mutate(
    variable = fct_relevel(
      variable,
      "psi",
      "tau",
      "phi",
      "gamma",
      "mu",
      "sigma",
      "xi",
      "Delta"
    )
  ) |>
  ggplot(aes(value, after_stat(scaled))) +
  geom_density(
    data = ~ rename(.x, md = model),
    aes(group = md),
    alpha = 0.2,
    fill = "grey60"
  ) +
  geom_density(
    aes(fill = model),
    alpha = 0.7
  ) +
  scale_x_continuous(
    expand = expansion(mult = 0.2),
    breaks = breaks_pretty(3)
  ) +
  scale_fill_brewer(
    palette = "Set1"
  ) +
  facet_grid(
    rows = vars(model),
    cols = vars(variable),
    scales = "free",
    labeller = label_parsed
  ) +
  theme(
    legend.position = "none",
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = "Comparing Smooth-Step Results With/Without Copula"
  )

ggsave(
  filename = "Figures/gevt/compare/comparison.png",
  width = 8,
  height = 0.5 * 8,
  scale = 1.5
)


p <- plot_dat |>
  mutate(
    variable = fct_relevel(
      variable,
      "psi",
      "tau",
      "phi",
      "gamma",
      "mu",
      "sigma",
      "xi",
      "Delta"
    )
  ) |>
  select(-type) |>
  pivot_wider(names_from = model) |>
  janitor::clean_names() |>
  mutate(
    variable2 = variable
  ) |>
  group_by(variable2) |>
  group_map(
    \(x, ...) {
      lower <- min(c(x$iid, x$copula))
      upper <- max(c(x$iid, x$copula))

      x |>
        ggplot(aes(copula, iid)) +
        geom_abline(
          intercept = 0,
          slope = 1,
          lty = 2
        ) +
        geom_point(
          alpha = 0.1
        ) +
        scale_x_continuous(
          guide = ggh4x::guide_axis_truncated(),
          breaks = scales::breaks_extended(6)
        ) +
        scale_y_continuous(
          guide = ggh4x::guide_axis_truncated(),
          breaks = scales::breaks_extended(6)
        ) +
        coord_cartesian(
          xlim = c(lower, upper),
          ylim = c(lower, upper)
        ) +
        labs(
          subtitle = latex2exp::TeX(
            str_c(
              "$",
              unique(x$variable),
              "$"
            )
          ),
          x = "Copula",
          y = "IID"
        )
    }
  ) |>
  wrap_plots(ncol = 4) +
  plot_annotation(
    title = "Comparing station-wise posterior medians from the IID and Copula models",
    subtitle = str_c(
      "Location, scale, shape and trend on the unconstrained (upper row) and constrained (lower row) scale"
    )
  )

ggsave(
  filename = "Figures/gevt/compare/iid_copula_compare.png",
  width = 8,
  height = 0.621 * 8,
  scale = 1.4
)
