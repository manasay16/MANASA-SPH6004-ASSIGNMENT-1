setwd("D:/")
csv_file_path <- "D:/sph6004_assignment1_data (1).csv"
akid <- read.csv(csv_file_path)

library(caret)
library(xgboost)
dim(akid)
head(akid)

akid$gender <- as.factor(akid$gender)
akid$race <- as.factor(akid$race)
akid$id <- as.factor(akid$id)
akid$aki <- as.factor(akid$aki)


table(akid$aki)


str(akid$aki)
summary(akid)
summary(akid$aki)

akidt <- na.roughfix(akid)
akidt$aki_binary <- as.factor(ifelse(akidt$aki %in% c(1, 2, 3), 1, 0))

predictors <- c("admission_age", "heart_rate_mean", "sbp_mean", "dbp_mean", "mbp_mean", "resp_rate_mean", "spo2_mean", 
                "hematocrit_max.1", "hemoglobin_min.1", "bicarbonate_min.1", "bicarbonate_max.1", "bun_min", "calcium_min.1",
                "calcium_max.1", "chloride_min.1", "chloride_max.1", "potassium_min.1", "potassium_max.1", "gcs_motor",
                "gcs_verbal", "gcs_eyes", "weight_admit")

# Subset the data 
X_df <- akidt[predictors]
y_df <- akidt$aki_binary

set.seed(123)
index <- createDataPartition(y_df, p = 0.7, list = FALSE)
train_data <- akidt[index, ]
test_data <- akidt[-index, ]


train_data$aki_numeric <- as.numeric(as.character(train_data$aki_binary))

params <- list(
  objective = "binary:logistic",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = table(train_data$aki_numeric)[1] / table(train_data$aki_numeric)[2]
)

train_data_numeric <- as.data.frame(sapply(train_data[predictors], as.numeric))

#XGBoost model 1
dtrain_binary <- xgb.DMatrix(data = as.matrix(train_data_numeric), label = train_data$aki_numeric)
xgbmbin1 <- xgboost(data = dtrain_binary, params = params, nrounds = 500, verbose = 1)


test_data$aki_binary <- as.factor(ifelse(test_data$aki %in% c(1, 2, 3), 1, 0))

test_data_numeric <- as.data.frame(sapply(test_data[predictors], as.numeric))

dtest_binary <- xgb.DMatrix(data = as.matrix(test_data_numeric))

# Make predictions on the test set
xgb_preds_binary <- predict(xgbmbin1, dtest_binary, iterationrange = c(1, 100))

# Convert predicted probabilities to class labels
xgb_preds_class_binary <- ifelse(xgb_preds_binary > 0.5, 1, 0)


cmb1 <- confusionMatrix(factor(xgb_preds_class_binary, levels = levels(test_data$aki_binary)), test_data$aki_binary)

# Calculate additional metrics
precisionb1 <- cmb1$byClass["Pos Pred Value"]
recallb1 <- cmb1$byClass["Sensitivity"]
f1_scoreb1 <- 2 * (precisionb1 * recallb1) / (precisionb1 + recallb1)

#SUMMARY OF XGBOOST MODEL 1
print("XGBOOST MODEL - 1")
summary(xgbmbin1)
print(cmb1)
cat("Precision:", precisionb1, "\n")
cat("Recall:", recallb1, "\n")
cat("F1-Score:", f1_scoreb1, "\n")

#...............................................................#
train_data$aki_numeric <- as.factor(train_data$aki_numeric)

params <- list(
  objective = "binary:logistic",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = table(train_data$aki_numeric)[1] / table(train_data$aki_numeric)[2]
)

train_data_numeric <- as.data.frame(sapply(train_data[predictors], as.numeric))

#preprocessing
preProc <- preProcess(train_data_numeric, method = c("center", "scale"))
train_data_scaled <- predict(preProc, train_data_numeric)
test_data_scaled <- predict(preProc, test_data_numeric)

# grid search
param_grid <- expand.grid(
  eta = c(0.01, 0.1, 0.2),
  max_depth = c(4, 6, 8),
  subsample = c(0.6, 0.8, 1.0),
  colsample_bytree = c(0.6, 0.8, 1.0)
)

train_data$aki_numeric <- as.numeric(as.character(train_data$aki_numeric))

#label scaling
max_label <- max(train_data$aki_numeric)
min_label <- min(train_data$aki_numeric)
range_label <- max_label - min_label

train_data$aki_numeric_scaled <- (train_data$aki_numeric - min_label) / range_label

# XGBoost model 2
dtrain_binary_scaled <- xgb.DMatrix(data = as.matrix(train_data_numeric), label = train_data$aki_numeric_scaled)
xgb_model_tuned <- xgboost(data = dtrain_binary_scaled, params = params, nrounds = 500, verbose = 1)

# Convert the test labels to 0 and 1
test_data$aki_numeric <- as.factor(ifelse(test_data$aki_binary == 0, 0, 1))
test_data_numeric <- as.data.frame(sapply(test_data[predictors], as.numeric))

dtest_binary_scaled <- xgb.DMatrix(data = as.matrix(test_data_numeric))
xgb_preds_binary_scaled <- predict(xgb_model_tuned, dtest_binary_scaled, iterationrange = c(1, 100))
xgb_preds_class_binary_scaled <- ifelse(xgb_preds_binary_scaled > 0.5, 1, 0)


cmb2 <- confusionMatrix(factor(xgb_preds_class_binary_scaled, levels = levels(test_data$aki_binary)), test_data$aki_binary)
precisionb2 <- cmb2$byClass["Pos Pred Value"]
recallb2 <- cmb2$byClass["Sensitivity"]
f1_scoreb2 <- 2 * (precisionb2 * recallb2) / (precisionb2 + recallb2)

#SUMMARY OF XGBOOST MODEL -2
print("XGBOOST MODEL-2 ")
print(cmb2)
cat("Precision:", precisionb2, "\n")
cat("Recall:", recallb2, "\n")
cat("F1-Score:", f1_scoreb2, "\n")

#...............................................................#

train_data$aki_numeric <- as.factor(train_data$aki_numeric)

# new parameter grid
param_grid2 <- expand.grid(
  nrounds = c(500),
  max_depth = c(6),
  eta = c(0.1),
  gamma = 0,
  colsample_bytree = c(0.8),
  min_child_weight = 1,
  subsample = c(0.8)
)

xgb_model_tuned2 <- train(
  x = train_data_scaled,
  y = train_data$aki_numeric,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE),
  tuneGrid = param_grid2,
  verbose = 1
)

test_data$aki_numeric <- as.factor(ifelse(test_data$aki_binary == 0, 0, 1))
test_data_numeric <- as.data.frame(sapply(test_data[predictors], as.numeric))

#DMatrix for test data
dtest_b3 <- xgb.DMatrix(data = as.matrix(test_data_numeric))

# Make predictions on the test set
xgb_preds_b3 <- predict(xgb_model_tuned2, newdata = test_data_scaled, type = "raw", ntree = 100)
xgb_preds_b3_numeric <- as.numeric(as.character(xgb_preds_b3))
xgb_preds_class_b3 <- ifelse(xgb_preds_b3_numeric > 0.5, 1, 0)


cmb3 <- confusionMatrix(factor(xgb_preds_class_b3, levels = levels(test_data$aki_binary)), test_data$aki_binary)
precisionb3 <- cmb3$byClass["Pos Pred Value"]
recallb3 <- cmb3$byClass["Sensitivity"]
f1_scoreb3 <- 2 * (precisionb3 * recallb3) / (precisionb3 + recallb3)

#SUMMARY OF XGBOOST MODEL -3
print("XGBOOST MODEL-3 ")
print(cmb3)
cat("Precision:", precisionb3, "\n")
cat("Recall:", recallb3, "\n")
cat("F1-Score:", f1_scoreb3, "\n")







