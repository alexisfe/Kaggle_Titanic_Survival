library(caret)
library(RANN)

set.seed(1234)

data_csv_link <- "train.csv"
test_csv_link <- "test.csv"

dep_var = "Survived"

data <- read.csv(data_csv_link)
eval_data <- read.csv(test_csv_link)

inTrain <- createDataPartition(data[, dep_var], p = 0.8)[[1]]
train <- data[inTrain,]
test <- data[-inTrain,]

#Select only predictor variables
train_pred <- train[, -which(names(train) == dep_var)]
test_pred <- test[, -which(names(test) == dep_var)]

#Cross-validate using 10 folds
train.ctrl <- trainControl(method="cv", number=10) 

#Preprocess data
preproc <- preProcess(train_pred, method=c("center", "scale", "nzv", "medianImpute", "pca"))

train_pred <- predict(preproc, train_pred)

test_pred <- predict(preproc, test_pred)

#Tune with grid
grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))

#Train model
model <- train(train[, dep_var] ~., method="rf", trControl=train.ctrl, tuneGrid=grid, data=train_pred)

#Evaluate model performance using test set
cm <- confusionMatrix(test[, dep_var], predict(model, test_pred))

cm$overall["Accuracy"]

eval_data_pred <- predict(preproc, eval_data)
predict(model, eval_data_pred)
