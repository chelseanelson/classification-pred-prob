# Classification Prediction Problem - Round 3
# Assess final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load testing and fitted data 
load(here("results/round_3/final_fit.rda")) # Baseline Random Forest version 2 
load(here("results/round_3/final_fit_2.rda")) # Baseline Boosted Tree version 1 
load(here("data/model_data/round_3/airbnb_test.rda"))
test_classification <- read_rds(here("data/test_classification.rds"))

# assessing models performance 

# Baseline Random Forest version 2 
airbnb_test_res <- bind_cols(airbnb_test, predict(final_fit, airbnb_test, type = "prob")) %>%
  select(id, host_is_superhost, .pred_TRUE, .pred_FALSE)

airbnb_model_metrics <- roc_auc(airbnb_test_res, host_is_superhost, .pred_TRUE) 

airbnb_model_metrics <- airbnb_model_metrics %>% rename(metric = .metric, estimate = .estimate) %>% select(-.estimator)

airbnb_curve <- roc_curve(airbnb_test_res, host_is_superhost, .pred_TRUE)

airbnb_curve_plot <- autoplot(airbnb_curve)

# Baseline Boosted Tree version 1
airbnb_test_res_2 <- bind_cols(airbnb_test, predict(final_fit_2, airbnb_test, type = "prob")) %>%
  select(id, host_is_superhost, .pred_TRUE, .pred_FALSE)

airbnb_model_metrics_2 <- roc_auc(airbnb_test_res_2, host_is_superhost, .pred_TRUE)

airbnb_model_metrics_2 <- airbnb_model_metrics_2 %>% rename(metric = .metric, estimate = .estimate) %>% select(-.estimator) 

airbnb_curve_2 <- roc_curve(airbnb_test_res_2, host_is_superhost, .pred_TRUE)

airbnb_curve_plot_2 <- autoplot(airbnb_curve_2)

# assessing models performance

# Baseline Random Forest version 2
airbnb_submission_3_2 <- bind_cols(test_classification, predict(final_fit, test_classification, type = "prob")) %>% select(id, .pred_TRUE) %>% rename(predicted = .pred_TRUE)

# Baseline Boosted Tree version 1 
airbnb_submission_3_3 <- bind_cols(test_classification, predict(final_fit_2, test_classification, type = "prob")) %>% select(id, .pred_TRUE) %>% rename(predicted = .pred_TRUE)

# save out results (plot, table)
write_rds(airbnb_curve_plot, file = here("results/round_3/airbnb_curve_plot.rds"))
write_rds(airbnb_model_metrics, file = here("results/round_3/airbnb_model_metrics.rds"))
write_csv(airbnb_submission_3_2, file = here("submissions/airbnb_submission_3_2.csv"))
write_rds(airbnb_curve_plot_2, file = here("results/round_3/airbnb_curve_plot_2.rds"))
write_rds(airbnb_model_metrics_2, file = here("results/round_3/airbnb_model_metrics_2.rds"))
write_csv(airbnb_submission_3_3, file = here("submissions/airbnb_submission_3_3.csv"))