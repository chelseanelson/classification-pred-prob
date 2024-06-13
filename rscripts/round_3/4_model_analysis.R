# Classification Prediction Problem - Round 3
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
load(here("results/round_3/fitted_tuned_models/tuned_bt_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_mars_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_rf_1.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_mars_2.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_rf_2.rda"))
load(here("results/round_3/fitted_tuned_models/tuned_bt_2.rda"))

# comparing sub-models ----
## Boosted Trees
bt_1_plot <- tuned_bt_1 %>% autoplot(metric = "roc_auc") # learn_rate: .029-.11,trees: 1200-1500, mtry 25-40 
bt_2_plot <- tuned_bt_2 %>% autoplot(metric = "roc_auc")
bt_best_1 <- show_best(tuned_bt_1, metric = "roc_auc")  
bt_best_2 <- show_best(tuned_bt_2, metric = "roc_auc")

## Random Forests 
rf_1_plot <- tuned_rf_1 %>% autoplot(metric = "roc_auc") #mtry 25 - 40, and trees between 1020 and 1350
rf_2_plot <- tuned_rf_2 %>% autoplot(metric = "roc_auc")
rf_best_1 <- show_best(tuned_rf_1, metric = "roc_auc")
rf_best_2 <- show_best(tuned_rf_2, metric = "roc_auc")

## MARS
mars_1_plot <- tuned_mars_1 %>% autoplot(metric = "roc_auc") # 1-10, num_terms 30-80
mars_2_plot <- tuned_mars_2 %>% autoplot(metric = "roc_auc") # 3, 5,10 and num_terms 20 - 55
mars_best_1 <- show_best(tuned_mars_1, metric = "roc_auc") 
mars_best_2 <- show_best(tuned_mars_2, metric = "roc_auc")

model_results_baseline <- as_workflow_set(mars_1 = tuned_mars_1,
                                 bt_1 = tuned_bt_1,
                                 rf_1 = tuned_rf_1)

# Best Model Currently: Boosted Tree 

model_results <- as_workflow_set(bt_1 = tuned_bt_1,
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
  distinct(wflow_id, .keep_all = TRUE) %>%
  arrange(std_err) %>%
  arrange(desc(mean)) %>% 
  select(wflow_id, .metric, mean, std_err, n) %>% 
  rename(metric = .metric)

model_accuracy_comparison <- model_results %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  slice_max(mean, by = wflow_id) %>% 
  distinct(wflow_id, .keep_all = TRUE) %>%
  arrange(std_err) %>%
  arrange(desc(mean)) %>% 
  select(wflow_id, .metric, mean, std_err, n) %>% 
  rename(metric = .metric)

# Best Model Overall: first version boosted tree model

# write out results (plots, tables)
write_rds(model_accuracy_comparison, file = here("results/round_3/fitted_tuned_models/model_accuracy_comparison.rds"))
