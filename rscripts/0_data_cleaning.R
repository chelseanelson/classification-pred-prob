# Classification Prediction Problem
# Initial data cleaning to load in correctly

# load packages
library(tidyverse)
library(here)

# cleaning the variables for the train and test datasets 
train_classification <- read_csv(here("data/train_classification.csv"),
                                 col_types = cols(id = col_character())) %>%
  mutate(
    host_is_superhost = factor(host_is_superhost, levels = c(TRUE, FALSE)),
    across(where(is.logical), factor),
    across(where(is.character), factor),
    host_response_rate = as.numeric(str_remove(host_response_rate, "%")),
    host_acceptance_rate = as.numeric(str_remove(host_acceptance_rate, "%")),
    number_vertifications = as.factor(str_count(host_verifications, "\\b\\w+\\b")),
    in_illinois = if_else(
      str_detect(host_location, "IL") | str_detect(host_location, "Illinois"),
      "Yes",
      "No"
    ))


test_classification <- read_csv(here("data/test_classification.csv"),
                                col_types = cols(id = col_character())) %>%
  mutate(
    across(where(is.logical), factor),
    across(where(is.character), factor),
    host_response_rate = as.numeric(str_remove(host_response_rate, "%")),
    host_acceptance_rate = as.numeric(str_remove(host_acceptance_rate, "%")),
    number_vertifications = as.factor(str_count(host_verifications, "\\b\\w+\\b")),
    in_illinois = if_else(
      str_detect(host_location, "IL") | str_detect(host_location, "Illinois"),
      "Yes",
      "No"
    ))

# write out rds
write_rds(train_classification, file = here("data/train_classification.rds"))
write_rds(test_classification, file = here("data/test_classification.rds"))