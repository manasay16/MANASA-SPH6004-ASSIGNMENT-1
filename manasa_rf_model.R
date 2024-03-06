#random forest model
getwd()
setwd("D:/")
csv_file_path <- "D:/sph6004_assignment1_data (1).csv"
akid <- read.csv(csv_file_path)

library(caret)
library(randomForest)
set.seed(123)

#converting variables to factors
akid$gender = as.factor(akid$gender)
akid$race = as.factor(akid$race)
akid$id = as.factor(akid$id)
akid$aki = as.factor(akid$aki)

# Imputing missing values
akidt = na.roughfix(akid)

# Creating binary outcome variable
akidt$aki_binary <- as.factor(ifelse(akidt$aki %in% c(1, 2, 3), 1, 0))

# Split data into training and testing sets
index <- createDataPartition(akidt$aki_binary, p = 0.8, list = FALSE)
train_data <- akidt[index, ]
test_data <- akidt[-index, ]

# Random Forest model
aki.modelrf <- randomForest(aki_binary ~ admission_age + heart_rate_mean + sbp_mean + dbp_mean + mbp_mean + resp_rate_mean + spo2_mean + hematocrit_max.1 + hemoglobin_min.1 + bicarbonate_min.1 + bicarbonate_max.1 + bun_min + calcium_min.1 + calcium_max.1 + chloride_min.1 + chloride_max.1 + potassium_min.1 + potassium_max.1 + gcs_motor + gcs_verbal + gcs_eyes + weight_admit , data = train_data, family = binomial())

# Predictions on the test set
prf <- predict(aki.modelrf, newdata = test_data)
conf_matrix_rf <- confusionMatrix(as.factor(prf), test_data$aki_binary)


# precision, recall, and F1 score
precision <- conf_matrix_rf$byClass["Pos Pred Value"]
recall <- conf_matrix_rf$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

#SUMMARY OF RANDOM FOREST MODEL
print("RANDOM FOREST MODEL")
summary(aki.modelrf)
print(conf_matrix_rf)
print(conf_matrix_rf)
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
