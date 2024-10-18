# https://bookdown.org/stanfordgsbsilab/ml-ci-tutorial/intro-ml.html


# Chapter 2 Introduction to Machine Learning


library(grf)
library(rpart)
library(glmnet)
library(splines)
library(lmtest)
library(MASS)
library(sandwich)
library(ggplot2)
library(reshape2)
library(reshape2)
library(stringr)

# Simulating data

# Sample size
n <- 500
x <- runif(n, -4, 4)
help(runif)
summary(x)

mu <- ifelse(x < 0, cos(2*x), 1-sin(x))
y <- mu + 1*rnorm(n)
summary(y)

data <- data.frame(x=x, y=y)
outcome <- "y"
covariates <- c("x")

plot(x, y, col="black", ylim=c(-4, 4), pch=21, bg="red", ylab="Outcome y", las=1)
lines(x[order(x)], mu[order(x)], col="black", lwd=3, type="l")
legend("bottomright", legend=c("Ground truth E[Y|X=x]", "Data"), cex=0.8, 
       lty=c(1, NA), col="black", pch=c(NA, 21), pt.bg=c(NA, "red"))

# 2.1 Key concepts

# Y_i = f(X_i) + ε_i

# Overfitting

subset <- 1:30
fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ", 10)"))
ols <- lm(fmla, data=data, subset=subset)
summary(ols)

x <- data[, covariates[1]]
x.grid <- seq(min(x), max(x), length.out=1000)
new.data <- data.frame(x.grid)
colnames(new.data) <- covariates[1]
summary(x.grid)

y.hat <- predict(ols, newdata = new.data)
summary(y.hat)

plot(data[subset, covariates[1]], data[subset, outcome], 
     pch=21, bg="red", xlab=covariates, ylab="Outcome y", las=1)
lines(x.grid, y.hat, col="green", lwd=2)
legend("bottomright", legend=c("Estimate", "Date"), col=c("green", "black"), 
       pch=c(NA, 21), pt.bg=c(NA, "red"), lty=c(2, NA), cex=0.8)

# Underfitting

subset <- 1:25
fmla <- formula(paste0(outcome, "~", covariates[1]))
summary(fmla)

ols <- lm(fmla, data[subset, ])
summary(ols)

x <- data[, covariates[1]]
x.grid <- seq(min(x), max(x), length.out=1000)
new.data <- data.frame(x.grid)
colnames(new.data) <- covariates[1]
summary(x.grid)

y.hat <- predict(ols, newdata = new.data)
summary(y.hat)

plot(data[subset, covariates[1]], data[subset, outcome], 
     pch=21, bg="red", xlab=covariates, ylab="Outcome y", las=1)
lines(x.grid, y.hat, col="green", lwd=2)
legend("bottomright", legend=c("Estimate", "Date"), col=c("green", "black"), 
       pch=c(NA, 21), pt.bg=c(NA, "red"), lty=c(2, NA), cex=0.8)


# bias-variance trade-off


poly.degree <- seq(3, 20)
train <- sample(1:n, 0.5*n)

mse.estimates <- lapply(poly.degree, function(q){
  fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ",", q,")"))
  ols <- lm(fmla, data=data[train,])
  
  y.hat.train <- predict(ols)
  y.train <- data[train, outcome]
  
  y.hat.test <- predict(ols, newdata=data[-train, ])
  y.test <- data[-train, outcome]
  
  data.frame(
    mse.train=mean((y.hat.train - y.train)^2),
    mse.train=mean((y.hat.test - y.test)^2))
})

mse.estimates <- do.call(rbind, mse.estimates)
matplot(poly.degree, mse.estimates, type="l", 
        ylab="MSE estimate", xlab="Polynomial degree", las="1")
legend("top", legend=c("Training", "Validation"), 
       bty="n", lty=1:2, col=1:2, cex=0.7)


# k-fold cross-validation


n.folds <- 5
poly.degree <- seq(4, 20)
indices <- split(seq(n), sort(seq(n) %% n.folds))

mse.estimates <- sapply(poly.degree, function(q){
  fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ",", q,")"))
  y.hat <- lapply(indices, function(fold.idx){
    ols <- lm(fmla, data=data[-fold.idx,])
    predict(ols, newdata=data[fold.idx,])
  })
  y.hat <- unname(unlist(y.hat))
  mean((y.hat - data[,outcome])^2)
})

plot(poly.degree, mse.estimates, ylab="MSE estimate", 
     xlab="Polynomial degree", type="l", lty=2, col=2, las = 1)
legend("top", legend=c("Cross-validated MSE"), bty="n", lty=2, col=2, cex=.7)


# using much more data


subset <- 1:n
fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ", 15)"))
ols <- lm(fmla, data=data, subset=subset)
summary(ols)

x <- data[,covariates[1]]
x.grid <- seq(min(x), max(x), length.out=1000)
colnames(new.data) <- covariates[1]
y.hat <- predict(ols, newdata = new.data)

plot(data[subset, covariates[1]], data[subset, outcome], pch=21, bg="red", 
     xlab=covariates[1], ylab="Outcome", las=1)
lines(x[order(x)], mu[order(x)], lwd=2, col="black")
lines(x.grid, y.hat, col="green", lwd=2)
legend("bottomright", lwd=2, lty=c(1,1), 
       col=c("black", "green"), legend=c("Ground truth", "Estimate"))


# 2.2 Common machine learning algorithms


data <- read.csv("https://docs.google.com/uc?id=1kNahFWMGUEB3Qz83s6rMf1l684MSqc_3&export=download")
dim(data)

outcome <- "LOGVALUE"

# covariates
true.covariates <- c('LOT','UNITSF','BUILT','BATHS','BEDRMS','DINING','METRO','CRACKS','REGION','METRO3','PHONE','KITCHEN','MOBILTYP','WINTEROVEN','WINTERKESP','WINTERELSP','WINTERWOOD','WINTERNONE','NEWC','DISH','WASH','DRY','NUNIT2','BURNER','COOK','OVEN','REFR','DENS','FAMRM','HALFB','KITCH','LIVING','OTHFN','RECRM','CLIMB','ELEV','DIRAC','PORCH','AIRSYS','WELL','WELDUS','STEAM','OARSYS')
p.true <- length(true.covariates)

# noise covariates added for didactic reasons
p.noise <- 20
noise.covariates <- paste0('noise', seq(p.noise))
covariates <- c(true.covariates, noise.covariates)
X.noise <- matrix(rnorm(n=nrow(data)*p.noise), nrow(data), p.noise)
colnames(X.noise) <- noise.covariates
data <- cbind(data, X.noise)
dim(data)

# sample size
n <- nrow(data)

# total number of covariates
p <- length(covariates)

round(cor(data[, covariates[1:8]]), 3)


# 2.2.1 Regularized linear models

# Lasso & Ridge
# Lasso penalizes the absolute value of slope coefficients(coef close to 0)
# Ridge penalizes the sum of squares of the slope coefficients(small coef)

# library(glmnet)

fmla <- formula(paste("~ 0 +", paste0(covariates, collapse="+")))
XX <- model.matrix(fmla, data)
dim(XX)
Y <- data[, outcome]

lasso <- cv.glmnet(x=XX, y=Y, family="gaussian", alpha=1)
help(cv.glmnet)

par(oma=c(0,0,3,0))
plot(lasso, las=1)
mtext("Number of Non-Zero Coefficients", side=3, line=3)

coef(lasso, s="lambda.min")[1:10,]

print(paste("Number of nonzero coefficients at optimial lambda:",
            lasso$nzero[which.min(lasso$cvm)], "out of", length(coef(lasso))))

y.hat <- predict(lasso, newx=XX, s="lambda.min", type="response")
mse.glmnet <- lasso$cvm[lasso$lambda == lasso$lambda.min]
print(paste("glmnet MSE estimate (k-fold corss-validation):", mse.glmnet))

pra(oma=c(0,0,3,0))
plot(lasso$glmnet.fit, xvar="lambda")
mtext("Number of Non-Zero Coefficients", side=3, line=3)




# Generating some data 
# y = 1 + 2*x1 + 3*x2 + noise, where corr(x1, x2) = .5
# note the sample size is very large -- this isn't solved by big data!
x <- mvrnorm(100000, mu=c(0,0), Sigma=diag(c(.5,.5)) + 1)
y <- 1 + 2*x[,1] + 3*x[,2] + rnorm(100000)
data.sim <- data.frame(x=x, y=y)
dim(data.sim)

print("Correct model")

lm(y ~ x.1 + x.2, data = data.sim)

print("Model with omitted variable bias")

lm(y ~ x.1, data = data.sim)


# post-lasso
# Using Lasso to select variables, and then using OLS


# prepare data
fmla <- formula(paste0(outcome, "~", paste0(covariates, collapse="+")))
XX <- model.matrix(fmla, data)[,-1]  # [,-1] drops the intercept
Y <- data[,outcome]

# fit ols, lasso and ridge models
ols <- lm(fmla, data)
lasso <- cv.glmnet(x=XX, y=Y, alpha=1.)  # alpha = 1 for lasso
ridge <- cv.glmnet(x=XX, y=Y, alpha=0.)  # alpha = 0 for ridge
summary(ols)

# retrieve ols, lasso and ridge coefficients
lambda.grid <- c(0, sort(lasso$lambda))
ols.coefs <- coef(ols)
lasso.coefs <- as.matrix(coef(lasso, s=lambda.grid))
ridge.coefs <- as.matrix(coef(ridge, s=lambda.grid))

# loop over lasso coefficients and re-fit OLS to get post-lasso coefficients
plasso.coefs <- apply(lasso.coefs, 2, function(beta) {
  
  # which slopes are non-zero
  non.zero <- which(beta[-1] != 0)  # [-1] excludes intercept
  
  # if there are any non zero coefficients, estimate OLS
  fmla <- formula(paste0(outcome, "~", paste0(c("1", covariates[non.zero]), collapse="+")))
  beta <- rep(0, ncol(XX) + 1)
  
  # populate post-lasso coefficients
  beta[c(1, non.zero + 1)] <- coef(lm(fmla, data))
  
  beta
})

selected <- "BATHS"
k <- which(rownames(lasso.coefs) == selected)
coefs <- cbind(postlasso=plasso.coefs[k,], lasso=lasso.coefs[k,], 
               ridge=ridge.coefs[k,], ols=ols.coefs[k])
matplot(lambda.grid, coefs, col=1:4, type="b", pch=20, lwd=2, las=1, 
        xlab="Lambda", ylab="Coefficient estimate")
abline(h = 0, lty="dashed", col="gray")
legend("bottomleft",
       legend = colnames(coefs),
       bty="n", col=1:4,   inset=c(.05, .05), lwd=2)

# other variables
covs <- which(covariates %in% c('UNITSF', 'BEDRMS',  'DINING'))
matplot(lambda.grid, t(lasso.coefs[covs+1,]), type="l", lwd=2, las=1, 
        xlab="Lambda", ylab="Coefficient estimate")
legend("topright", legend = covariates[covs], bty="n", 
       col=1:p,  lty=1:p, inset=c(.05, .05), lwd=2, cex=.6)



# Fixing lambda. This choice is not very important; 
# the same occurs any intermediate lambda value.
selected.lambda <- lasso$lambda.min
n.folds <- 10
foldid <- (seq(n) %% n.folds) + 1
coefs <- sapply(seq(n.folds), function(k) {
  lasso.fold <- glmnet(XX[foldid == k,], Y[foldid == k])
  as.matrix(coef(lasso.fold, s=selected.lambda))
})
heatmap(1*(coefs != 0), Rowv = NA, Colv = NA, cexCol = 1, scale="none", 
        col=gray(c(1,0)), margins = c(3, 1), xlab="Fold", 
        labRow=c("Intercept", covariates), main="Non-zero coefficient estimates")


# data-driven subgroups


num.groups <- 4
n.folds <- 5
foldid <- (seq(n) %% n.folds) + 1
fmla <- formula(paste("~ 0 +", paste0("bs(", covariates, ", df=3)", collapse="+")))

XX <- model.matrix(fmla, data)
Y <- data[, outcome]

lasso <- cv.glmnet(x=XX, y=Y, foldid=foldid, keep=T)
y.hat <- predict(lasso, newx=XX, s="lambda.min")

ranking <- lapply(seq(n.folds), function(i){
  y.hat.cross.val <- y.hat[foldid==i]
  qs <- quantile(y.hat.cross.val, probs=seq(0,1, length.out=num.groups + 1))
  cut(y.hat.cross.val, breaks=qs, labels=seq(num.groups))
})

ranking <- factor(do.call(c, ranking))

# Estimate expected covariate per subgroup
avg.covariate.per.ranking <- mapply(function(x.col) {
  fmla <- formula(paste0(x.col, "~ 0 + ranking"))
  ols <- lm(fmla, data=transform(data, ranking=ranking))
  t(lmtest::coeftest(ols, vcov=vcovHC(ols, "HC2"))[, 1:2])
}, covariates, SIMPLIFY = FALSE)

avg.covariate.per.ranking[1:2]


# the average covariate per group along with each standard errors


df <- mapply(function(covariate) {
  # Looping over covariate names
  # Compute average covariate value per ranking (with correct standard errors)
  fmla <- formula(paste0(covariate, "~ 0 + ranking"))
  ols <- lm(fmla, data=transform(data, ranking=ranking))
  ols.res <- coeftest(ols, vcov=vcovHC(ols, "HC2"))
  
  # Retrieve results
  avg <- ols.res[,1]
  stderr <- ols.res[,2]
  
  # Tally up results
  data.frame(covariate, avg, stderr, ranking=paste0("G", seq(num.groups)), 
             # Used for coloring
             scaling=pnorm((avg - mean(avg))/sd(avg)), 
             # We will order based on how much variation is 'explain' by the averages
             # relative to the total variation of the covariate in the data
             variation=sd(avg) / sd(data[,covariate]),
             # String to print in each cell in heatmap below
             # Note: depending on the scaling of your covariates, 
             # you may have to tweak these formatting parameters a little.
             labels=paste0(formatC(avg), "\n", " (", formatC(stderr, digits = 2, width = 2), ")"))
}, covariates, SIMPLIFY = FALSE)
df <- do.call(rbind, df)

# a small optional trick to ensure heatmap will be in decreasing order of 'variation'
df$covariate <- reorder(df$covariate, order(df$variation))
df <- df[order(df$variation, decreasing=TRUE),]

# plot heatmap
ggplot(df[1:(9*num.groups),]) +  # showing on the first few results (ordered by 'variation')
  aes(ranking, covariate) +
  geom_tile(aes(fill = scaling)) + 
  geom_text(aes(label = labels), size=3) +  # 'size' controls the fontsize inside cell
  scale_fill_gradient(low = "#E1BE6A", high = "#40B0A6") +
  ggtitle(paste0("Average covariate values within group (based on prediction ranking)")) +
  theme_minimal() + 
  ylab("") + xlab("") +
  theme(plot.title = element_text(size = 10, face = "bold"),
        legend.position="bottom")


# 2.2.2 Decision trees


# Fit tree without pruning first
fmla <- formula(paste(outcome, "~", paste(covariates, collapse=" + ")))
tree <- rpart(fmla, data=data, cp=0, method="anova")  
# use method="class" for classification
plot(tree, uniform=TRUE)
# prune the tree
plotcp(tree)


# Retrieves the optimal parameter
cp.min <- which.min(tree$cptable[,"xerror"]) # minimum error
cp.idx <- which(tree$cptable[,"xerror"] - tree$cptable[cp.min,"xerror"] < tree$cptable[,"xstd"])[1]  # at most one std. error from minimum error
cp.best <- tree$cptable[cp.idx,"CP"]

# Prune the tree
pruned.tree <- prune(tree, cp=cp.best)

plot(pruned.tree, uniform=T, margin=0.05)
text(pruned.tree, cex=0.7)


# Retrieve predictions from pruned tree
y.hat <- predict(pruned.tree)

# Compute mse for pruned tree (using cross-validated predictions)
mse.tree <- mean((xpred.rpart(tree)[,cp.idx] - data[,outcome])^2, na.rm=TRUE)
print(paste("Tree MSE estimate (cross-validated):", mse.tree))


y.hat <- predict(pruned.tree)

# Number of leaves should equal the number of distinct prediction values.
# This should be okay for most applications, but if an exact answer is needed use
# predict.rpart.leaves from package treeCluster
num.leaves <- length(unique(y.hat))

# Leaf membership, ordered by increasing prediction value
leaf <- factor(y.hat, ordered = TRUE, labels = seq(num.leaves))

# Looping over covariates
avg.covariate.per.leaf <- mapply(function(covariate) {
  
  # Coefficients on linear regression of covariate on leaf 
  #  are the average covariate value in each leaf.
  # covariate ~ leaf.1 + ... + leaf.L 
  fmla <- formula(paste0(covariate, "~ 0 + leaf"))
  ols <- lm(fmla, data=transform(data, leaf=leaf))
  
  # Heteroskedasticity-robust standard errors
  t(coeftest(ols, vcov=vcovHC(ols, "HC2"))[,1:2])
}, covariates, SIMPLIFY = FALSE)

print(avg.covariate.per.leaf[1:2])  # Showing only first few



df <- mapply(function(covariate) {
  # Looping over covariate names
  # Compute average covariate value per ranking (with correct standard errors)
  fmla <- formula(paste0(covariate, "~ 0 + leaf"))
  ols <- lm(fmla, data=transform(data, leaf=leaf))
  ols.res <- coeftest(ols, vcov=vcovHC(ols, "HC2"))
  
  # Retrieve results
  avg <- ols.res[,1]
  stderr <- ols.res[,2]
  
  # Tally up results
  data.frame(covariate, avg, stderr, 
             ranking=factor(seq(num.leaves)), 
             # Used for coloring
             scaling=pnorm((avg - mean(avg))/sd(avg)), 
             # We will order based on how much variation is 'explain' by the averages
             # relative to the total variation of the covariate in the data
             variation=sd(avg) / sd(data[,covariate]),
             # String to print in each cell in heatmap below
             # Note: depending on the scaling of your covariates, 
             # you may have to tweak these  formatting parameters a little.
             labels=paste0(formatC(avg),"\n(", formatC(stderr, digits = 2, width = 2), ")"))
}, covariates, SIMPLIFY = FALSE)
df <- do.call(rbind, df)

# a small optional trick to ensure heatmap will be in decreasing order of 'variation'
df$covariate <- reorder(df$covariate, order(df$variation))
df <- df[order(df$variation, decreasing=TRUE),]

# plot heatmap
ggplot(df[1:(8*num.leaves),]) +  # showing on the first few results (ordered by 'variation')
  aes(ranking, covariate) +
  geom_tile(aes(fill = scaling)) + 
  geom_text(aes(label = labels), size=2.5) +  # 'size' controls the fontsize inside cell
  scale_fill_gradient(low = "#E1BE6A", high = "#40B0A6") +
  ggtitle(paste0("Average covariate values within leaf")) +
  theme_minimal() + 
  ylab("") + xlab("Leaf (ordered by prediction, low to high)") +
  labs(fill="Normalized\nvariation") +
  theme(plot.title = element_text(size = 12, face = "bold", hjust = .5),
        axis.title.x = element_text(size=9),
        legend.title = element_text(hjust = .5, size=9))



# 2.2.3 Forests


X <- data[,covariates]
Y <- data[,outcome]

# Fitting the forest
# We'll use few trees for speed here. 
# In a practical application please use a higher number of trees.
forest <- regression_forest(X=X, Y=Y, num.trees=200)  

# There usually isn't a lot of benefit in tuning forest parameters, 
# but the next code does so automatically (expect longer training times)
# forest <- regression_forest(X=X, Y=Y, tune.parameters="all")

# Retrieving forest predictions
y.hat <- predict(forest)$predictions

# Evaluation (out-of-bag mse)
mse.oob <- mean(predict(forest)$debiased.error)
print(paste("Forest MSE (out-of-bag):", mse.oob))


var.imp <- variable_importance(forest)
names(var.imp) <- covariates
sort(var.imp, decreasing = TRUE)[1:10] # showing only first few



