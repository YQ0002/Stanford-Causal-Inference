rm(list = ls())
gc()

library(lmtest)
library(sandwich)
library(grf)
library(glmnet)
library(splines)
library(ggplot2)
library(reshape2)
library(RColorBrewer)

data <- read.csv('D:/R/ML-based casual inference/synthetic_data.csv')
str(data)

# schoolid
# S3 Student’s self-reported expectations for success in the future
# C1 Student race/ethnicity          --Categorical
# C2 Student gender                  --Categorical
# C3 Student first-generation status --Categorical
# XC School Urbanicity               --Categorical
# X1 School-level mean of students’ fixed mindsets
# X2 School achievement level, as measured by test scores and college preparation
# X3 School racial/ethnic minority composition
# X4 School poverty concentration
# X5 School size

# Y Post-treatment outcome, a continuous measure of achievement
# Z Treatment, i.e., receipt of the intervention

treatment <- "Z"
outcome <- "Y"
covariates <- c("schoolid", "S3", "C1", "C2", "C3", "XC", "X1", "X2", "X3", "X4", "X5")

hist(data$Y)
summary(data$Y)

# Figure 1: Propensity scores
fmla <- as.formula(paste0("~", paste0("bs(", covariates, ", df=3)", collapse="+")))
Z <- data[,treatment]
Y <- data[,outcome]
X <- model.matrix(fmla, data)

set.seed(123)
logit <- cv.glmnet(x=X, y=Z, family="binomial")
e.hat <- predict(logit, X, s = "lambda.min", type="response")

hist(e.hat)
boxplot(e.hat~data$S3)


# Estimating treatment effects with causal forests.
str(data)
school.id <- data[, "schoolid"]

Y.forest <- regression_forest(X, Y, clusters = school.id)
Y.hat <- predict(Y.forest)$predictions
Z.forest <- regression_forest(X, Z, clusters = school.id)
Z.hat <- predict(Z.forest)$predictions

cf.raw <- causal_forest(X, Y, Z, 
                        Y.hat = Y.hat, 
                        W.hat = Z.hat, 
                        clusters = school.id)
# help("causal_forest") # Notice: W.hat
varimp <- variable_importance(cf.raw)
selected.idx <- which(varimp > mean(varimp))
length(selected.idx)
length(varimp)

set.seed(123)
cf <- causal_forest(X[, selected.idx], Y, Z, 
                    Y.hat = Y.hat, 
                    W.hat = Z.hat, 
                    clusters = school.id,
                    min.node.size = 50, # samples_per_cluster
                    tune.parameters = "all") # TRUE
tau.hat <- predict(cf)$predictions
hist(tau.hat)

ATE <- average_treatment_effect(cf)
ATE
# 0.2542582 +/- 1.96*0.0171661 

# Compare regions with high and low estimated CATEs
high_effect <- tau.hat > median(tau.hat)
ate.high <- average_treatment_effect(cf, subset = high_effect)
ate.low <- average_treatment_effect(cf, subset = !high_effect)


# mean.forest.prediction: y_hat ~ y_i
# differential.forest.prediction: cov(t_i_hat - t_i))
test_calibration(cf)
help("test_calibration")


boxplot(tau.hat~data[,"X1"])
help("boxplot")


summary(data[,"X1"])
X1 <- round(data[,"X1"], 0) 
head(X1)
boxplot(tau.hat~X1)

# names(varimp) <- covariates
# barplot(varimp, horiz = T)


dr.score <- tau.hat + Z/Z.hat *

