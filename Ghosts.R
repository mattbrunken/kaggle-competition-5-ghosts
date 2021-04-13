### Ghost classification

## libraries
library(xgboost)
library(tidyverse)


## read in data

train <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/train.csv')

test <- read.csv('/Users/matthewbrunken/Winter2021/Kaggle/competition5/test.csv')

test$type <- sample(train$type, nrow(test), replace = TRUE)

full <- bind_rows(train,test,.id = 'Set')

full$Set <- ifelse(full$Set == 1, 'train', 'test')

# make 'type' and 'color' a factor
#train$type <- as.factor(train$type)
#train$color <- as.factor(train$color)
#train$color <- as.integer(train$color)
#test$color <- as.factor(test$color)
#test$color <- as.integer(test$color)

full$color <- as.numeric(as.factor(full$color))

## convert labels

Type = full$type
label = as.integer(as.factor(full$type))-1
full$type = NULL

# train/test split
train.index = as.numeric(rownames(full %>% filter(Set == 'train')))
train.data = as.matrix(full[which(full$Set == 'train'), 2:6])
train.label = label[train.index]
test.data = as.matrix(full[-train.index,2:6])
test.label = label[-train.index]


## create xgboost matrices
xgb.train = xgb.DMatrix(data = train.data,label = train.label)

xgb.test = xgb.DMatrix(data=test.data,label=test.label)


# Define the parameters for multinomial classification


num_class = length(levels(factor(Type)))
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class 
)

# Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=10000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=0
)

# Review the final model and results
xgb.fit

# predict new values
xgb.pred = predict(xgb.fit,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = unique(Type)

# Use the predicted label with the highest probability
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])


# create submission
submission <- cbind(1:529, xgb.pred[,4])
names(submission) <- c('id', 'type')

# write to .csv file
write.csv(submission, '/Users/matthewbrunken/Winter2021/Kaggle/competition5/submission.csv', 
          row.names = FALSE)
