# Load required libraries
library(glmnet)
library(doParallel)
library(pROC)

# Enable parallel processing
registerDoParallel(cores = detectCores() - 1)

# Handle missing values in training data (remove or impute)
train_data <- na.omit(train_data)  # Remove rows with NAs (alternative: use imputation)

# Convert training data to matrix format
train_matrix <- as.matrix(train_data[, -which(names(train_data) == "TARGET")])
y_train <- train_data$TARGET

# Check for class imbalance
print(table(y_train))

# Tune lambda using cross-validation
cv_model <- cv.glmnet(train_matrix, y_train, family = "binomial", alpha = 1, parallel = TRUE)
best_lambda <- cv_model$lambda.min

# Train LASSO Logistic Regression
final_model <- glmnet(train_matrix, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

# Extract selected features correctly
coef_matrix <- as.matrix(coef(final_model))
selected_vars <- rownames(coef_matrix)[coef_matrix[, 1] != 0]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]

# Define new formula for logistic regression
formula <- as.formula(paste("TARGET ~", paste(selected_vars, collapse = " + ")))

# Train final logistic regression model
final_glm <- glm(formula, data = train_data, family = binomial)

# Make predictions on test set
pred_prob <- predict(final_glm, newdata = test_data, type = "response")

# Apply 0.5 threshold to classify
test_data$predicted <- ifelse(pred_prob > 0.5, 1, 0)

# Evaluate model performance
conf_matrix <- table(test_data$TARGET, test_data$predicted)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

# Print results
print(paste("Model Accuracy:", round(accuracy, 4)))
print("Confusion Matrix:")
print(conf_matrix)
