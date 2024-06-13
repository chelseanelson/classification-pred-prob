## Classification Prediction Problem - Round 1 Tuned and Fitted Models 

Directory contains results from all scripts used for fitting/training and/or tuning models. 

`tuned_knn_1.rda`: Results from tuning the kitchen sink k-nearest neighbor model/workflow on resamples

`tuned_mars_1.rda`: Results from tuning the kitchen sink mars model/workflow on resamples

`tuned_nnet_1.rda`: Results and data from tuning the kitchen sink nerual network model/workflow on resamples 

`fit_logistic_1.rda`: Results and data from fitting the baseline logistic regression model/workflow on resamples 

`fit_logistic_2.rda`: Results and data from fitting the feature engineered logistic regression model/workflow on resamples 

`fit_nbayes.rda`: Results and data from fitting the baseline naive bayes model/workflow on resamples 

`tuned_rf_1.rda`: Results from tuning the kitchen sink random forest model/workflow on resamples 

`tuned_bt_1.rda`: Results from tuning the kitchen sink boosted tree model/workflow on resamples 

`tuned_rf_2.rda`: Results from tuning the feature engineered random forest model/workflow on resamples 

`tuned_bt_2.rda`: Results from tuning the feature engineered boosted tree model/workflow on resamples 

`tuned_mars_2.rda`: Results from tuning the feature engineered mars model/workflow on resamples 

`tuned_nnet_2.rda`: Results from tuning the feature engineered neural network model/workflow on resamples

`tuned_knn_2.rda`: Results from tuning the feature engineered k-nearest neighbors model/workflow on resamples 
`model_accuracy_comparison.rds` : table showcasing how all the fitted/tuned models type compared in terms of their performance looking specifically at their `roc_auc` values
