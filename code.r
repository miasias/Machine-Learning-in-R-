################################################################################
#This code uses the Cardiovascular Disease Dataset to predict the risk of a person having a heart disease.
################################################################################

#Load the required Packages
library(tidyverse)
library(paradox)
library(mlr3tuning)
library(data.table)
library(ggplot2)
library(patchwork)
library(readr)
library(caTools)
library(dplyr)
library(parallel)
library(future)
library(rpart)
library(rpart.plot)
library(mlr3viz)
library(iml)
library(ggcorrplot)
library(mlr3measures)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)

#Read the uploaded .csv dataset
data <- read_csv("Cardiovascular_Disease_Dataset.csv")

#########################
#-----Preprocessing-----#
#########################

# Inspect the data
summary(data)
head(data)
str(data)
View(data)

# Check for missing values in the dataset
na_count <- sapply(data, function(j) { sum(is.na(j)) })
print(na_count)

# Basic variable plotting to further get insight into the data
# Plot the target variable
targ <- ggplot(data, aes(x = factor(target))) +
  coord_cartesian(ylim = c(0, 800)) +
  geom_bar(fill = "darkgreen", color = "black") +
  labs(
    x = "Target (0 = No Heart Disease, 1 = Heart Disease)",
    y = "Number of Patients",
    title = "Distribution of the Target Variable"
  ) +
  theme_minimal()
print(targ)

# Plot the gender of the patients
gend <- ggplot(data, aes(x = factor(gender))) +
  coord_cartesian(ylim = c(0, 800)) +
  geom_bar(fill = "darkgreen", color = "black") +
  labs(
    x = "Gender (0 = Female, 1 = Male)",
    y = "Number of Patients",
    title = "Distribution of Gender"
  ) +
  theme_minimal()
print(gend)

# Create a combined plot to analyse Data distribution
largeplot <- targ + gend
print(largeplot)

# Plot age distribution
age <-gendage <- ggplot(data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "darkgreen", color = "white") +
  labs(
    title = "Distribution of Age",
    x = "Age (years)",
    y = "Number of Patients"
  ) +
  theme_minimal()
print(age)

# Plot the slope variable
slope <- ggplot(data, aes(x = factor(slope))) +
  geom_bar(fill = "darkgreen", color = "black") +
  labs(
    title = "Distribution of the Slope of the Peak Exercise ST Segment",
    x = "Slope (1 = Upsloping, 2 = Flat, 3 = Downsloping)",
    y = "Number of Patients"
  ) +
  theme_minimal()
print(slope)

# Because the plotting showed data points out of scope we further investigate the variables
# to check for missing values coded as 0s and values that cannot be true
sum(data$age==0)
sum(!(data$gender %in% c(0, 1)))
sum(!(data$chestpain %in% c(0, 1, 2, 3)))
sum(data$restingBP==0)
sum(data$serumcholestrol==0)
sum(!(data$fastingbloodsugar %in% c(0, 1)))
sum(!(data$restingrelectro %in% c(0, 1, 2)))
sum(data$maxheartrate==0)
sum(!(data$exerciseangia %in% c(0, 1)))
sum(!(data$slope %in% c(1, 2, 3)))
sum(!(data$noofmajorvessels %in% c(0, 1, 2, 3)))

# Because we have missing values in slope and serumcholestrol we further check if there
# is any overlap between them
missing <- subset(data, serumcholestrol == 0 & slope == 0)
view(missing)

# Because there are 180 missing values in the column "slope" we drop this variable from the dataset
data <- subset(data, select = -slope)

#Correlation
numeric_vars <- c("age", "restingBP", "serumcholestrol", "maxheartrate", "oldpeak", "noofmajorvessels")
data_corr <- subset(data, select = numeric_vars)

#Notiz: use = "complete.obs" eig unnötig
corr <- round(cor(data_corr, use = "complete.obs"), 2)
ggcorrplot(corr, lab = TRUE)

##############################################
#-----Data split and vairable imputation-----#
##############################################

# Convert 0 to NA in serumcholestrol
data$serumcholestrol[data$serumcholestrol == 0] <- NA

#Split data into Train / Test Sets 
set.seed(1234)
train <- data %>% sample_frac(., .66)
test <- anti_join(data, train, by = 'patientid')

# Calculate the median and removing the NA values
median_train <- median(train$serumcholestrol, na.rm = TRUE)

# Impute the missing values for train / test data with the median of the training data
train$serumcholestrol[is.na(train$serumcholestrol)] <- median_train
test$serumcholestrol[is.na(test$serumcholestrol)] <- median_train

# Test if there are no 0s in serumcholestrol column
sum(train$serumcholestrol==0)
sum(test$serumcholestrol==0)

# Drop ID column for modeling
train <- subset(train, select = -c(patientid))
test <- subset(test, select = -c(patientid))

##########################
#-----Model training-----#
##########################

#-----Baseline Dummy Model-----

# Majority class (0 or 1)
train$target %>% table()

dummy_class <- 1

# Predictions for baseline
predict_dummy <- rep(dummy_class, nrow(test))

# Evaluate baseline
confusion_matrix <- table(test$target, predict_dummy)

dummy_accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
dummy_mmce <- 1 - sum(diag(confusion_matrix))/sum(confusion_matrix)

cat('Dummy MMCE:', (dummy_mmce),'-', 'Accuracy:', dummy_accuracy)

#-----Logistic Regression-----

logistic_model <- glm(target ~ ., 
                      data = train,
                      family = "binomial"
                      )
summary(logistic_model)

# Predict on test data
predict_reg <- predict(logistic_model, 
                       newdata = test,
                       type = "response"
                       )

predict_reg <- ifelse(predict_reg > 0.5, 1, 0)

# Evaluation
confusion_matrix <- table(test$target, predict_reg)

logreg_accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
logreg_mmce <- 1 - sum(diag(confusion_matrix))/sum(confusion_matrix)

cat('LogReg MMCE:', logreg_mmce,'-', 'Accuracy:', logreg_accuracy)

# Comparison: Dummy vs. Logistic Regression
tab <- matrix(rep(2, times = 4), ncol = 2, byrow = TRUE)
colnames(tab) <- c('MMCE', 'ACC')
rownames(tab) <- c('Baseline', 'Logistic Regression')

# Measures Baseline Model
tab[1,1] = dummy_mmce
tab[1,2] = dummy_accuracy

# Measures Logistic Regression Model
tab[2,1] = logreg_mmce
tab[2,2] = logreg_accuracy

tab
rowSums(tab)

#-----Decision Tree-----
# Convert target variable into a factor
data$target <- as.factor(data$target)

# 
task_data <- TaskClassif$new(
  id = "data",
  backend = data,
  target = "target",
  positive = "1"
)

# Check the features 
task_data$col_roles$feature

# Remove patientid from the role as feature
task_data$set_col_roles("patientid", remove_from = "feature")

# set split
set.seed(1234)
splits <- partition(task_data, ratio = 0.66)

# Define a pipeline object for imputation
imputer = po("imputemedian")

#Train Decision Tree
dtree <- lrn("classif.rpart")

# Define that %>>% is used from the mlr3pipelines package and not from caTools
graph <- mlr3pipelines::`%>>%`(imputer, dtree)
tree_imp <- as_learner(graph)

tree_imp$train(task_data, splits$train)
prediction_tree <- tree_imp$predict(task_data, splits$test)
prediction_tree$confusion

# Decision Tree plot with title
rpart.plot(tree_imp$model$classif.rpart$model, main = "Decision Tree for Heart Disease Prediction")

# Decision Tree Performance
mes <- msrs(c("classif.ce", "classif.acc"))
tree_perf <- prediction_tree$score(mes)
tree_perf

#-----Random Forest-----
rf <- lrn("classif.ranger", num.trees = 500, predict_type = "prob")
graphrf<- mlr3pipelines::`%>>%`(imputer, rf)
rf_imp <- as_learner(graphrf)
rf_imp$train(task_data, splits$train)
prediction_rf <- rf_imp$predict(task_data, splits$test)
prediction_rf$confusion

rf_perf <- prediction_rf$score(mes)
rf_perf

#Cross Validation

rdesc <- rsmp("cv", folds = 10)
res <- resample(task = task_data, learner = rf_imp, resampling = rdesc)
res$aggregate(mes)

acc <- res$score(msr("classif.ce"))
resamplings <- acc[, .(iteration, classif.ce)]
mean(resamplings$classif.ce)
sd(resamplings$classif.ce)

##########################
#Hyperparameter Tuning & Parallel Computing
##########################

#Core Detection
detectCores()
cores = detectCores() - 1

#Parallelization
plan("multisession", workers = cores)

#Define tuning Parameters

rf_ps <- ps(
  `classif.ranger.mtry`          = p_int(lower = 1, upper = 11),
  `classif.ranger.min.node.size` = p_int(lower = 1, upper = 30))


res_inner = rsmp("cv", folds = 5)
mes_inner = msr("classif.mcc")
terminator = trm("evals", n_evals = 50)
tuner = mlr3tuning::tnr("random_search")

#Create Graphlearner that performs tuning automatically
rf_at = AutoTuner$new(
  learner = rf_imp,
  resampling = res_inner,
  measure = mes_inner,
  search_space = rf_ps,
  terminator = terminator,
  tuner = tuner 
)

#Evaluation of Predicitve Performance
res_outer = rsmp("cv", folds = 10)

nested_res = resample(
  task = task_data,
  learner = rf_at,
  resampling = res_outer
)

plan("sequential")

nested_res$aggregate()
autoplot(nested_res)

#-----Benchmark-----
plan("multisession", workers = cores)

logreg <- lrn("classif.log_reg")
tree <- lrn("classif.rpart", keep_model = TRUE)
rf <- lrn("classif.ranger", num.trees = 500, predict_type = "prob")
rf_tuned <- rf_at
baseline = lrn("classif.featureless")

# rf_tuned is not imputed again because it is built in
graphlogreg<- mlr3pipelines::`%>>%`(imputer, logreg)
logreg_imp_bench = as_learner(graphlogreg)

graphtree<- mlr3pipelines::`%>>%`(imputer, tree)
tree_imp_bench = as_learner(graphtree)

graphrfbench<- mlr3pipelines::`%>>%`(imputer, rf)
rf_imp_bench = as_learner(graphrfbench)

graphbase<- mlr3pipelines::`%>>%`(imputer, baseline)
baseline_imp_bench = as_learner(graphbase)


#Construct a Benchmak Grid
design_class = benchmark_grid(
  tasks = task_data,
  learners = list( baseline_imp_bench, logreg_imp_bench, tree_imp_bench, rf_imp_bench, rf_tuned), 
  resamplings = rsmp("cv", folds = 10)
  )

#Run the Benchmark
bm_class = benchmark(design_class)
plan("sequential")

#Benchmark Evaluation
mes_class = msrs(c("classif.ce", "classif.acc", "classif.precision"))
bmr_class = bm_class$aggregate(mes_class)
bmr_class[, c(4,7:9)]
autoplot(bm_class, measure = msr("classif.ce"))

#-----Variable Importance-----

# Make a copy for variable importance
data_vi <- data

# Impute serumcholestrol in THIS copy using the median of all non-missing values
med_sc <- median(data_vi$serumcholestrol, na.rm = TRUE)
data_vi$serumcholestrol[is.na(data_vi$serumcholestrol)] <- med_sc

data_vi$patientid <- NULL

task_vi <- TaskClassif$new(
  id      = "heart_vi",
  backend = data_vi,
  target  = "target",
  positive = "1"
)

logreg_vi <- lrn("classif.log_reg")
tree_vi   <- lrn("classif.rpart", keep_model = TRUE)
rf_vi     <- lrn("classif.ranger", num.trees = 500, predict_type = "prob")

logreg_model = logreg_vi$train(task_vi)
tree_model   = tree_vi$train(task_vi)
rf_model     = rf_vi$train(task_vi)

# Logistic Regression - Beta Coefficients
beta_weights <- coef(logreg_model$model)

beta_df <- data.frame(
  Feature     = names(beta_weights),
  Coefficient = as.numeric(beta_weights),
  stringsAsFactors = FALSE
)

# Drop intercept for later plotting
beta_df <- subset(beta_df, Feature != "(Intercept)")

betas_plot <- ggplot(beta_df,
                     aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "darkgreen", color = "black") +
  coord_flip() +
  labs(
    title = "Logistic Regression Coefficients",
    x = "Feature",
    y = "Coefficient (Beta)"
  ) +
  theme_minimal()

print(betas_plot)

# Decision tree plot
rpart.plot(tree_model$model, main = "Decision Tree for Heart Disease Prediction")


#RF 
y <- data_vi$target

# X without target
X <- subset(data_vi, select = -c(target))

# Create iml predictor
mod <- Predictor$new(rf_model, data = X, y = y)

# Compute importance
importance = FeatureImp$new(mod, loss = "ce", n.repetitions = 10)
importance$plot()

##########################
#Final Comparison
##########################

mes_class = msrs(c("classif.ce", "classif.acc", "classif.precision"))
bmr_class = bm_class$aggregate(mes_class)
bmr_class[, c(4,7:9)]
autoplot(bm_class, measure = msr("classif.ce")) 
