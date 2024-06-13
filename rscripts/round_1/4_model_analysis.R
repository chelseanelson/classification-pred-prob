# Classification Prediction Problem - Round 1 
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
load(here("results/round_1/fitted_tuned_models/fit_logistic_1.rda"))
load(here("results/round_1/fitted_tuned_models/fit_nbayes.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_bt_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_knn_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_mars_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_nnet_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_rf_1.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_knn_2.rda"))
load(here("results/round_1/fitted_tuned_models/fit_logistic_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_mars_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_nnet_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_bt_2.rda"))
load(here("results/round_1/fitted_tuned_models/tuned_rf_2.rda"))

# comparing sub-models ----
## Logistic Regressions
logistic_best_1 <- show_best(fit_logistic_1, metric = "roc_auc")
logistic_best_2 <- show_best(fit_logistic_2, metric = "roc_auc")

## Naive Bayes 
nbayes_best <- show_best(fit_nbayes, metric = "roc_auc")

## Neutral Networks
nnet_1_plot <- tuned_nnet_1 %>% autoplot(metric = "roc_auc") # try hidden_units() > 5 
nnet_2_plot <- tuned_nnet_2 %>% autoplot(metric = "roc_auc")
nnet_best_1 <- show_best(tuned_nnet_1, metric = "roc_auc")
nnet_best_2 <- show_best(tuned_nnet_2, metric = "roc_auc")

## K-Nearest Neighbors 
knn_1_plot <- tuned_knn_1 %>% autoplot(metric = "roc_auc") # increase nearest neighbors > 8 
knn_2_plot <- tuned_knn_2 %>% autoplot(metric = "roc_auc")
knn_best_1 <- show_best(tuned_knn_1, metric = "roc_auc")
knn_best_2 <- show_best(tuned_knn_2, metric = "roc_auc")

## Boosted Trees
bt_1_plot <- tuned_bt_1 %>% autoplot(metric = "roc_auc") # mtry should look between 14 and 24, min_n = 1, trees between 900 and 1500, learn rate is good 
bt_2_plot <- tuned_bt_2 %>% autoplot(metric = "roc_auc") # boosted tree got worse but these are good parameters, related to the recipes maybe 
bt_best_1 <- show_best(tuned_bt_1, metric = "roc_auc") 
bt_best_2 <- show_best(tuned_bt_2, metric = "roc_auc") 

# Random Forests 
rf_1_plot <- tuned_rf_1 %>% autoplot(metric = "roc_auc") # try higher trees, and 15 < mtry < 35 
rf_2_plot <- tuned_rf_2 %>% autoplot(metric = "roc_auc") # random forest got worse, related to the recipes maybe?
rf_best_1 <- show_best(tuned_rf_1, metric = "roc_auc")
rf_best_2 <- show_best(tuned_rf_2, metric = "roc_auc")

## MARS
mars_1_plot <- tuned_mars_1 %>% autoplot(metric = "roc_auc") # higher interactions better > 3, also more model terms > 15
mars_2_plot <- tuned_mars_2 %>% autoplot(metric = "roc_auc")
mars_best_1 <- show_best(tuned_mars_1, metric = "roc_auc") 
mars_best_2 <- show_best(tuned_mars_2, metric = "roc_auc")

model_results_baseline <- as_workflow_set(nbayes = fit_folds_nbayes,
                                 logisitic_1 = fit_folds_logistic_1,
                                 nnet_1 = tuned_nnet_1,
                                 knn_1 = tuned_knn_1,
                                 bt_1 = tuned_bt_1,
                                 rf_1 = tuned_rf_1,
                                 mars_1 = tuned_mars_1
)

# Best Model Currently: Boosted Tree Model

model_results <- as_workflow_set(nbayes = fit_nbayes,
                                 logisitic_1 = fit_logistic_1,
                                 logistic_2 = fit_logistic_2,
                                 nnet_1 = tuned_nnet_1,
                                 nnet_2 = tuned_nnet_2,
                                 knn_1 = tuned_knn_1,
                                 knn_2 = tuned_knn_2,
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

# Best Model Overall: Baseline Boosted Trees

# write out results (plots, tables)
write_rds(model_accuracy_comparison, file = here("results/round_1/fitted_tuned_models/model_accuracy_comparison.rds"))
