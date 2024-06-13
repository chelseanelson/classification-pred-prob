# Classification Prediction Problem - Round 2
# Analysis of tuned and trained models (comparison)
# Main Assessment Metric : ROC_AUC

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load in tuned models 
load(here("results/round_2/fitted_tuned_models/tuned_bt_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_svm_poly_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_svm_radial_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_mars_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_rf_1.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_svm_radial_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_mars_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_bt_2.rda"))
load(here("results/round_2/fitted_tuned_models/tuned_rf_2.rda"))

# comparing sub-models ----
## SVM Polynomial
svm_poly_1_plot <- tuned_svm_poly_1 %>% autoplot(metric = "roc_auc") # did the worse not moving to second stage
svm_poly_best_1 <- show_best(tuned_svm_poly_1, metric = "roc_auc")

## SVM Radial
svm_radial_1_plot <- tuned_svm_radial_1 %>% autoplot(metric = "roc_auc") # look at cost more deeply 2 and up maybe to 40
svm_radial_2_plot <- tuned_svm_radial_2 %>% autoplot(metric = "roc_auc")
svm_radial_best_1 <- show_best(tuned_svm_radial_1, metric = "roc_auc")
svm_radial_best_2 <- show_best(tuned_svm_radial_2, metric = "roc_auc")

## Boosted Trees
bt_1_plot <- tuned_bt_1 %>% autoplot(metric = "roc_auc") # trees between 1000 and 1200, and mtry between 20-26
bt_2_plot <- tuned_bt_2 %>% autoplot(metric = "roc_auc")
bt_best_1 <- show_best(tuned_bt_1, metric = "roc_auc")
bt_best_2 <- show_best(tuned_bt_2, metric = "roc_auc")

## Random Forests 
rf_1_plot <- tuned_rf_1 %>% autoplot(metric = "roc_auc") # mtry should be around 10-20, trees between 1000-1400
rf_2_plot <- tuned_rf_2 %>% autoplot(metric = "roc_auc")
rf_best_1 <- show_best(tuned_rf_1, metric = "roc_auc")
rf_best_2 <- show_best(tuned_rf_2, metric = "roc_auc")

## MARS
mars_1_plot <- tuned_mars_1 %>% autoplot(metric = "roc_auc") # prod_degree of 6 seems to be the best, num_terms 60 to 80
mars_2_plot <- tuned_mars_2 %>% autoplot(metric = "roc_auc") # more components is better, so maybe step_pca is not worth it ?
mars_best_1 <- show_best(tuned_mars_1, metric = "roc_auc") 
mars_best_2 <- show_best(tuned_mars_2, metric = "roc_auc")


model_results_baseline <- as_workflow_set(mars_1 = tuned_mars_1,
                                 svm_poly_1 = tuned_svm_poly_1,
                                 svm_radial_1 = tuned_svm_radial_1,
                                 bt_1 = tuned_bt_1,
                                 rf_1 = tuned_rf_1
)

# Best Model Currently:  Baseline Boosted Tree Model

model_results <- as_workflow_set(svm_radial_1 = tuned_svm_radial_1,
                                 svm_radial_2 = tuned_svm_radial_2,
                                 svm_poly_1 = tuned_svm_poly_1,
                                 bt_1 = tuned_bt_1,
                                 bt_2 = tuned_bt_2,
                                 rf_1 = tuned_rf_1,
                                 rf_2 = tuned_rf_2,
                                 mars_1 = tuned_mars_1,
                                 mars_2 = tuned_mars_2
)

model_accuracy_comparison_baseline <- model_results_baseline %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  slice_max(mean, by = wflow_id) %>% 
  arrange(std_err) %>%
  arrange(desc(mean)) %>% 
  select(wflow_id, .metric, mean, std_err, n) %>% 
  rename(metric = .metric)

model_accuracy_comparison <- model_results %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  slice_max(mean, by = wflow_id) %>% 
  arrange(std_err) %>%
  arrange(desc(mean)) %>% 
  select(wflow_id, .metric, mean, std_err, n) %>% 
  rename(metric = .metric)

# Best Model Overall: Baseline Boosted Tree Model

# write out results (plots, tables)
write_rds(model_accuracy_comparison, file = here("results/round_2/fitted_tuned_models/model_accuracy_comparison.rds"))
