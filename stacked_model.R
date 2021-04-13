## Stacked model

## libraries
library(dplyr)
library(caret)
library(xgboost)
library(gbm)

## read in data

train <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/train.csv')

test <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/test.csv')

full <- bind_rows(train, test, .id = 'Set')
full$Set <- ifelse(full$Set == 1, 'train', 'test')

## read in probability datasets

nn <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/nn_Probs_65acc.csv')
xgb_tree <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/xgbTree_probs (1).csv')
knn <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/Probs_KNN.csv')
svm <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/probs_svm.csv')
gbm <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/probs_gbm.csv')
mp <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/multilayerperceptron.csv')
rf <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/classification_submission_rf.csv')

# make the ID columns the same for everyone
names(nn)[4] <- 'ID'
names(mp)[1] <- 'ID'
names(full)[2] <- 'ID'

## stick on probs to full dataset

merged <- left_join(full, nn, by = 'ID') %>%
  left_join(., xgb_tree, by = 'ID') %>% 
  left_join(., knn, by = 'ID') %>% 
  left_join(., svm, by = 'ID') %>% 
  left_join(., gbm, by = 'ID') %>% 
  left_join(., mp, by = 'ID') %>% 
  left_join(., rf, by = 'ID')


## pre-processing
pp <- predict(preProcess(merged[,-c(1,2)], method = 'pca'), merged[,-c(1,2)])
pp$ID <- merged$ID
pp$Set <- merged$Set

## split out to train and test
pp_train <- pp %>% filter(Set == 'train')
pp_train$type <- as.factor(pp_train$type)
pp_test <- pp %>% filter(Set == 'test')

## create tuning grid for xgboost
grid <- expand.grid(n.trees = c(1000,1500), 
                    interaction.depth=c(1:3), 
                    shrinkage=c(0.01,0.05,0.1), 
                    n.minobsinnode=c(20))

##  create train control
ctrl <- trainControl(method = "repeatedcv",
                     number = 5, 
                     repeats = 2, allowParallel = T)

## fit xgb model
unwantedoutput <- capture.output(GBMModel <- caret::train(type~.,data = pp_train[, -c(13,14)],
                  method = "gbm", trControl = ctrl, tuneGrid = grid))

## look at model
GBMModel

## convert color to numeric factor
pp_test$color <- as.numeric(as.factor(pp_test$color))

## predict new values
preds <- xgb_model %>% stats::predict(pp_test %>% select(-type, -Set, -ID))

## put preds into dataframe to match up with ID
pp_test$type <- preds

# create submission
submission <- pp_test %>% select(ID, type) %>% as.data.frame()

# export .csv
write.csv(submission, '/Users/matthewbrunken/Winter2021/Kaggle/competition5/Stacked/submission2.csv',
          row.names = FALSE)


