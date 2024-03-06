#SVM model
library(caret)
library(randomForest)
library(e1071)
getwd()
set.seed(123)

getwd()
setwd("D:/")
csv_file_path <- "D:/sph6004_assignment1_data (1).csv"
akid <- read.csv(csv_file_path)


akid$gender = as.factor(akid$gender)
akid$race = as.factor(akid$race)
akid$id = as.factor(akid$id)
akid$aki = as.factor(akid$aki)

dim(akid)
head(akid)
table(akid$aki)
str(akid$aki)


summary(akid)
summary(akid$aki)

# Impute missing values
akidt = na.roughfix(akid)

# Create binary outcome variable
akidt$aki_binary <- as.factor(ifelse(akidt$aki %in% c(1, 2, 3), 1, 0))

# Split data into training and testing sets
index <- createDataPartition(akidt$aki_binary, p = 0.8, list = FALSE)
train_data <- akidt[index, ]
test_data <- akidt[-index, ]

# SVM model
aki.modelsvm <- svm(aki_binary ~ admission_age + heart_rate_mean + sbp_mean + dbp_mean + mbp_mean + resp_rate_mean + spo2_mean + hematocrit_max.1 + hemoglobin_min.1 + bicarbonate_min.1 + bicarbonate_max.1 + bun_min + calcium_min.1 + calcium_max.1 + chloride_min.1 + chloride_max.1 + potassium_min.1 + potassium_max.1 + gcs_motor + gcs_verbal + gcs_eyes + weight_admit , data = train_data, family = binomial())

# Predictions on the test set
prsvm <- predict(aki.modelsvm, newdata = test_data)
conf_matrix_svm <- confusionMatrix(as.factor(prsvm), test_data$aki_binary)

# Calculate precision, recall, and F1 score
precisionsvm <- conf_matrix_svm$byClass["Pos Pred Value"]
recallsvm <- conf_matrix_svm$byClass["Sensitivity"]
f1_scoresvm <- 2 * (precisionsvm * recallsvm) / (precisionsvm + recallsvm)

# Summary of SVM model
print("SVM MODEL")
summary(aki.modelsvm)
print(conf_matrix_svm)
cat("Precision:", precisionsvm, "\n")
cat("Recall:", recallsvm, "\n")
cat("F1 Score:", f1_scoresvm, "\n")
