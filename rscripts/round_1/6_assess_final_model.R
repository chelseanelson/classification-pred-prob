# Classification Prediction Problem - Round 1
# Assess final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# load testing and fitted data 
load(here("results/round_1/final_fit.rda"))
load(here("data/model_data/round_1/airbnb_test.rda"))
test_classification <- read_rds(here("data/test_classification.rds"))

# assessing models performance 
airbnb_test_res <- bind_cols(airbnb_test, predict(final_fit, airbnb_test, type = "prob")) %>%
  select(id, host_is_superhost, .pred_TRUE, .pred_FALSE)

airbnb_test_res <- bind_cols(airbnb_test, predict(final_fit, airbnb_test)) %>%
  select(.pred_class) %>% bind_cols(airbnb_test_res) 

airbnb_test_res <- airbnb_test_res %>% relocate(.pred_class, .before = .pred_TRUE)

airbnb_model_metrics <- roc_auc(airbnb_test_res, host_is_superhost, .pred_TRUE) 
airbnb_curve <- roc_curve(airbnb_test_res, host_is_superhost, .pred_TRUE)

airbnb_curve_plot <- autoplot(airbnb_curve)

# assessing models performance
airbnb_submission_1 <- bind_cols(test_classification, predict(final_fit, test_classification, type = "prob")) %>% select(id, .pred_TRUE) %>% rename(predicted = .pred_TRUE)

# save out results (plot, table)
write_rds(airbnb_curve_plot, file = here("results/round_1/airbnb_curve_plot.rds"))
write_rds(airbnb_model_metrics, file = here("results/round_1/airbnb_model_metrics.rds"))
write_csv(airbnb_submission_1, file = here("submissions/airbnb_submission_1_2.csv"))

# thoughts for next round and submission score 
# gave me a score of .95288
# try step_impute_median instead of the mean for the baseline recipe 
# see if step_impute_knn for the feature engineered one
# create a tree specific recipe because they don't need corr and yeojohnson, i think
# everything else stays the same within the baseline and feature engineered recipes 
# move on with mars, random forest, and boosted tree in addition to adding svm poly and radial 
# could also look at mars again and test increasing the degree of interaction, as well as the making the range from num_terms start about 20 and go to 50?