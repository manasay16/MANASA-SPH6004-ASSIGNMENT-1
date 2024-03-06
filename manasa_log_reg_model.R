#Logistic regression model
setwd("D:/")
library(glm2)
library(randomForest)
library(caret)

csv_file_path <- "D:/sph6004_assignment1_data (1).csv"
akid <- read.csv(csv_file_path)

dim(akid)
head(akid)

akid$gender <- as.factor(akid$gender)
akid$race <- as.factor(akid$race)
akid$id <- as.factor(akid$id)

str(akid)
summary(akid)

# Impute missing values
akid <- na.roughfix(akid)

# Logistic regression model (without data split)
akid$aki_binary <- as.factor(ifelse(akid$aki %in% c(1, 2, 3), 1, 0))

set.seed(123) 
index <- createDataPartition(akid$aki_binary, p = 0.8, list = FALSE)
train_data <- akid[index, ]
test_data <- akid[-index, ]

#Logistic regression model with all variables
aki.modellr <- glm(aki_binary ~  gender + admission_age + heart_rate_mean + sbp_mean + dbp_mean + mbp_mean + resp_rate_mean + 
                     spo2_mean +  glucose_mean + hematocrit_min.1 + hematocrit_max.1 + hemoglobin_min.1 + hemoglobin_max.1 + 
                     platelets_min + platelets_max + wbc_min + wbc_max + aniongap_min + aniongap_max + bicarbonate_min.1 + 
                     bicarbonate_max.1 + bun_min + bun_max + calcium_min.1 + calcium_max.1 + chloride_min.1 + chloride_max.1 
                   + glucose_min.2 + glucose_max.2 + sodium_min.1 + sodium_max.1 + potassium_min.1 +  potassium_max.1 + 
                     inr_min + inr_max + pt_min + pt_max + gcs_min + gcs_motor + gcs_verbal + gcs_eyes + gcs_unable + 
                     weight_admit , data = train_data, family = binomial())

# Predictions on the test set
prlr <- predict(aki.modellr, newdata = test_data, type = "response")

# Set levels for the predictions
prlr <- as.factor(ifelse(prlr > 0.5, 1, 0))
levels(prlr) <- levels(test_data$aki_binary)

# Confusion Matrix
cmlr <- confusionMatrix(data = prlr, reference = test_data$aki_binary)


# Calculating precision, recall, and F1 score
prlr <- cmlr$byClass["Pos Pred Value"]
recalllr <- cmlr$byClass["Sensitivity"]
f1_scorelr <- 2 * (prlr * recalllr) / (prlr + recalllr)

#SUMMARY OF LOGISTIC REGRESSION MODEL 1
print("Logistic regression model -1")
summary(aki.modellr)
print(cmlr)
cat("Precision:", prlr, "\n")
cat("Recall:", recalllr, "\n")
cat("F1 Score:", f1_scorelr, "\n")

#logistic regression model 2
aki.modellr2 <- glm(aki_binary ~  admission_age + heart_rate_mean + sbp_mean + dbp_mean + mbp_mean + resp_rate_mean + spo2_mean + hematocrit_max.1 + hemoglobin_min.1 + bicarbonate_min.1 + bicarbonate_max.1 + bun_min + calcium_min.1 + calcium_max.1 + chloride_min.1 + chloride_max.1 + potassium_min.1 + potassium_max.1 + gcs_motor + gcs_verbal + gcs_eyes + weight_admit , data = train_data, family = binomial())


# Predictions on the test set
prlr2 <- predict(aki.modellr2, newdata = test_data, type = "response")

# Set levels for the predictions
prlr2 <- as.factor(ifelse(prlr2 > 0.5, 1, 0))
levels(prlr2) <- levels(test_data$aki_binary)

# Confusion Matrix
cmlr2 <- confusionMatrix(data = prlr2, reference = test_data$aki_binary)

# Calculate precision, recall, and F1 score
prlr2 <- cmlr2$byClass["Pos Pred Value"]
recalllr2 <- cmlr2$byClass["Sensitivity"]
f1_scorelr2 <- 2 * (prlr2 * recalllr2) / (prlr2 + recalllr2)

#SUMMARY OF LOGISTIC REGRESSION MODEL 2
print("Logistic regression model -2")
summary(aki.modellr2)
print(cmlr2)
cat("Precision:", prlr2, "\n")
cat("Recall:", recalllr2, "\n")
cat("F1 Score:", f1_scorelr2, "\n")

