## Classification Prediction Problem - Round 3 Rscripts 

`1_initial_setup.R`: R script containing the initial setup of the data

`2_recipes.R`: R script containing the making of the model recipes 

`3_tune_bt-1.R`: R script containing the making and tuning of the version 1 kitchen sink boosted tree model 

`3_tune_bt-2.R`: R script containing the making and tuning of the version 2 kitchen sink boosted tree model

`3_tune_rf-1.R`: R script containing the making and tuning of the version 1 kitchen sink random forest model

`3_tune_rf-2.R`: R script containing the making and tuning of the version 2 kitchen sink boosted tree model

`3_tune_mars-1.R`: R script containing the making and tuning of the version 1 kitchen sink mars model

`3_tune_mars-2.R`: R script containing the making and tuning of the version 2 kitchen sink mars model

`4_model_analysis.R`: R script containing the full usage and testing of my predictive models, look at each resampling and tuned models' performance, selecting the best based on the accuracy performance metric.

`5_train_final_model.R`: R script containing the training of the first final model choice on the full training set

`5_train_final_model.R`: R script containing the training of the second final model choice on the full training set

`6_assess_final_model.R` R script containing the assessment of the final model choices on the full testing set
 
## Usage

If you want to explore or run the code, I would recommend starting with 
`rscripts/1_initial_setup.R` to understand the distribution and layout of the 
original data and then running in order after that in order to see the 
development and creation of the different types of models. I would finish off
by looking at `rscripts/4_model_analysis.R` through 
`rscripts/6_assess_final_model.R` to see how each model performs, 
looking at which has the most accuracy. Otherwise, all of the associated model 
data, fitted data, tables of the performance analyses, and the recipes in which
I created can be found in their respective rscripts. 
