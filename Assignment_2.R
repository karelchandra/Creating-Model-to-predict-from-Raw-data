getwd()
setwd("/Users/karelchandra/Desktop/SEM 1 2022/FIT 3152 Data Analytics/Assignment_2")

library(tree)
library(e1071)
library(ROCR)
library(randomForest)
library(adabag)
library(rpart)

# Create Data Set
rm(list = ls())
WAUS <- read.csv("WarmerTomorrow2022.csv", stringsAsFactors = TRUE)
L <- as.data.frame(c(1:49))
set.seed(30373867) # Your Student ID is the random seed
L <- L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS <- WAUS[(WAUS$Location %in% L),]
WAUS <- WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows


# Q1 
# 1.) Find the proportion of warmer days
Warm <- sum(WAUS$WarmerTomorrow == 1, na.rm = TRUE)
Proportion <- (Warm/nrow(WAUS))*100
print(Proportion)
# 2.) Find the predictors
summary(WAUS)
apply(WAUS, 2, sd, na.rm = TRUE)
fitted = lm(WAUS$WarmerTomorrow ~ WAUS$Day + WAUS$Month + WAUS$Year + 
              WAUS$Location + WAUS$MinTemp + WAUS$MaxTemp + WAUS$Rainfall +
              WAUS$Evaporation + WAUS$Sunshine + WAUS$WindGustDir + WAUS$WindGustSpeed +
              WAUS$WindDir9am + WAUS$WindDir3pm + WAUS$WindSpeed9am + WAUS$WindSpeed3pm +
              WAUS$Humidity9am + WAUS$Humidity3pm + WAUS$Pressure9am + WAUS$Pressure3pm +
              WAUS$Cloud9am + WAUS$Cloud3pm + WAUS$Temp9am + WAUS$Temp3pm)
summary(fitted)

# Q2
# Remove NA value so that model can work
WAUS <- na.omit(WAUS)

# Q3
set.seed(30373867) #Student ID as random seed
train.row = sample(1:nrow(WAUS), 0.7*nrow(WAUS))
WAUS.train = WAUS[train.row,]
WAUS.test = WAUS[-train.row,]
WAUS.train$WarmerTomorrow <- as.factor(WAUS.train$WarmerTomorrow)
WAUS.test$WarmerTomorrow <- as.factor(WAUS.test$WarmerTomorrow)

# Q4
#  Calculate a descision tree
WAUS.tree = tree(WarmerTomorrow ~., data = WAUS.train)
plot(WAUS.tree)
text(WAUS.tree, pretty = 0)
# Calculate naive bayes
WAUS.bayes = naiveBayes(as.factor(WarmerTomorrow) ~. , data = WAUS.train)

# Bagging
WAUS.bag <- bagging(WarmerTomorrow ~. , data = WAUS.train, mfinal=5) 

#Boosting
WAUS.Boost <- boosting(WarmerTomorrow ~. , data = WAUS.train, mfinal=10) 

# Random Forest
WAUS.rf <- randomForest(WarmerTomorrow ~. , data = WAUS.train, na.action = na.exclude) 


# Q5
# Decision Tree
# do predictions as classes and draw a table
WAUS.predtree = predict(WAUS.tree, WAUS.test, type = "class") 
t1=table(Predicted_Class = WAUS.predtree, Actual_Class = WAUS.test$WarmerTomorrow)
accuracy_dt <- sum(t1[1], t1[4]) / sum(t1[1:4])
cat("\n#Decision Tree Confusion\n")
print(t1)

# Naive Bayes
WAUS.predbayes = predict(WAUS.bayes, WAUS.test)
t2=table(Predicted_Class = WAUS.predbayes, Actual_Class = WAUS.test$WarmerTomorrow) 
accuracy_nb <- sum(t2[1], t2[4]) / sum(t2[1:4])
cat("\n#NaiveBayes Confusion\n")
print(t2)

# Bagging
WAUSpred.bag <- predict.bagging(WAUS.bag, WAUS.test)
accuracy_b <- sum(WAUSpred.bag$confusion[1], WAUSpred.bag$confusion[4]) / sum(WAUSpred.bag$confusion[1:4])
cat("\n#Bagging Confusion\n")
print(WAUSpred.bag$confusion)

# Boosting
WAUSpred.boost <- predict.boosting(WAUS.Boost, newdata=WAUS.test)
accuracy_bs <- sum(WAUSpred.boost$confusion[1], WAUSpred.boost$confusion[4]) / sum(WAUSpred.boost$confusion[1:4])
cat("\n#Boosting Confusion\n")
print(WAUSpred.boost$confusion)

# Random Forest
WAUSpredrf <- predict(WAUS.rf, WAUS.test)
t3=table(Predicted_Class = WAUSpredrf, Actual_Class = WAUS.test$WarmerTomorrow) 
accuracy_rf <- sum(t3[1], t3[4]) / sum(t3[1:4])
cat("\n#Random Forest Confusion\n")
print(t3)


# Q6
#Decision tree
# do predictions as probabilities and draw ROCs
WAUS.pred.tree = predict(WAUS.tree, WAUS.test, type = "vector")
WAUSpred_dt <- prediction( WAUS.pred.tree[,2], WAUS.test$WarmerTomorrow)
WAUSperf_dt <- performance(WAUSpred_dt,"tpr","fpr")
plot(WAUSperf_dt)
# calculate and print auc
cauc_dt = performance(WAUSpred_dt, "auc")
print(as.numeric(cauc_dt@y.values))
abline(0,1)

# Naive Bayes
# outputs as confidence levels
WAUSpred.bayes = predict(WAUS.bayes, WAUS.test, type = 'raw') 
WAUSpred_nb <- prediction( WAUSpred.bayes[,2], WAUS.test$WarmerTomorrow) 
WAUSperf_nb <- performance(WAUSpred_nb,"tpr","fpr") 
plot(WAUSperf_nb, add = TRUE, col = "blueviolet")
# calculate and print auc
cauc_nb = performance(WAUSpred_nb, "auc")
print(as.numeric(cauc_nb@y.values))

# Bagging
WAUSBagpred <- prediction( WAUSpred.bag$prob[,2], WAUS.test$WarmerTomorrow) 
WAUSBagperf <- performance(WAUSBagpred,"tpr","fpr") 
plot(WAUSBagperf, add=TRUE, col = "blue")
# calculate and print auc
cauc_bag = performance(WAUSBagpred, "auc")
print(as.numeric(cauc_bag@y.values))

# Boosting
WAUSBoostpred <- prediction( WAUSpred.boost$prob[,2], WAUS.test$WarmerTomorrow) 
WAUSBoostperf <- performance(WAUSBoostpred,"tpr","fpr") 
plot(WAUSBoostperf, add=TRUE, col = "red")
# calculate and print auc
cauc_boo = performance(WAUSBoostpred, "auc")
print(as.numeric(cauc_boo@y.values))

# Random Forest
WAUSpred.rf <- predict(WAUS.rf, WAUS.test, type="prob")
# JCpred.rf
WAUSFpred <- prediction( WAUSpred.rf[,2], WAUS.test$WarmerTomorrow) 
WAUSFperf <- performance(WAUSFpred,"tpr","fpr") 
plot(WAUSFperf, add=TRUE, col = "darkgreen")
# calculate and print auc
cauc_rf = performance(WAUSFpred, "auc")
print(as.numeric(cauc_rf@y.values))

legend(x = "bottomright", legend = c("DT", "NB", "BAG", "BOOST", "RF"),
       lty = c(1,1,1,1,1),
       col = c("black", "blueviolet", "blue", "red", "darkgreen"))

# Q7
Classifiers <- c("Decision Tree", "Naive Bayes", "Bagging", "Boosting", "Random Forest")
Accuracy <- c(accuracy_dt, accuracy_nb, accuracy_b, accuracy_bs, accuracy_rf)
AUC <- c(as.numeric(cauc_dt@y.values), as.numeric(cauc_nb@y.values), as.numeric(cauc_bag@y.values), as.numeric(cauc_boo@y.values), as.numeric(cauc_rf@y.values))
df_c <- data.frame(Classifiers, Accuracy, AUC)
print(df_c)

# Q8
# Decision Tree
Waus_dt <- tree(WarmerTomorrow ~ ., WAUS.train)
summary(Waus_dt)

# Bagging
WAUS.bag$importance

# Boosting
WAUS.Boost$importance

# Random Forest
print(WAUS.rf$importance)

# Q9
# Exclude not important predictors
WAUS.train_b <- subset(WAUS.train, select = -c(Cloud3pm, Cloud9am,
                                               Day, Humidity9am,
                                               Location, MinTemp,
                                               Month, Rainfall,
                                               WindGustSpeed, Year, 
                                               WindDir9am, WindDir3pm, WindGustDir))
WAUS.test_b <- subset(WAUS.test, select = -c(Cloud3pm, Cloud9am,
                                             Day, Humidity9am,
                                             Location, MinTemp,
                                             Month, Rainfall,
                                             WindGustSpeed, Year,
                                             WindDir9am, WindDir3pm, WindGustDir))
WAUS.train_b$WarmerTomorrow <- as.factor(WAUS.train_b$WarmerTomorrow)
Waus_dt_b<- tree(WarmerTomorrow ~ ., WAUS.train_b)
summary(Waus_dt_b)
plot(Waus_dt_b)
text(Waus_dt_b, pretty = 0)

# do predictions as classes and draw a table
Waus_dt_b.predtree = predict(Waus_dt_b, WAUS.test_b, type = "class") 
t1=table(Actual_Class = WAUS.test_b$WarmerTomorrow, Predicted_Class =Waus_dt_b.predtree) 
cat("\n#Decsion Tree Confusion Better\n")
print(t1)
accuracy_dt_b <- sum(t1[1], t1[4]) / sum(t1[1:4])
print(accuracy_dt_b)
# do predictions as probabilities and draw ROC 
Waus_dt_b.pred.tree = predict(Waus_dt_b, WAUS.test_b, type = "vector") 
Waus_dt_bDpred <- prediction( Waus_dt_b.pred.tree[,2], WAUS.test_b$WarmerTomorrow) 
Waus_dt_bDperf <- performance(Waus_dt_bDpred,"tpr","fpr")
plot(Waus_dt_bDperf, col = "red")
plot(WAUSperf_dt, add=TRUE, col = "blue")
abline(0,1)
# calculate and print auc
cauc_dt_b = performance(Waus_dt_bDpred, "auc")
print(as.numeric(cauc_dt_b@y.values))

Classifiers <- c("Decision Tree", "Better Version Decision Tree")
Accuracy <- c(accuracy_dt,accuracy_dt_b)
AUC <- c(as.numeric(cauc_dt@y.values), as.numeric(cauc_dt_b@y.values))
df_c9 <- data.frame(Classifiers, Accuracy, AUC)
print(df_c9)

# Q10
# Exclude not important predictors
WAUS.train_cv <- subset(WAUS.train, select = -c(Location, Rainfall, Month, Cloud9am))
WAUS.test_cv <- subset(WAUS.test, select = -c(Location, Rainfall, Month, Cloud9am))
# Random Forest
WAUS.rf_cv <- randomForest(WarmerTomorrow ~. , data = WAUS.train_cv, na.action = na.exclude) 
Waus.pred = predict(WAUS.rf_cv, WAUS.test_cv)
tP <- table(actual = WAUS.test_cv$WarmerTomorrow, predicted = Waus.pred)
accuracy_rf_cv <- sum(tP[1], tP[4]) / sum(tP[1:4])
cat("\n#Random Forest Confusion\n")
print(tP)
print(accuracy_rf_cv)

Waus.pred_cv = predict(WAUS.rf_cv, WAUS.test_cv, type="prob")
WAUSpred_cv <- prediction(Waus.pred_cv[,2], WAUS.test_cv$WarmerTomorrow)
WAUSperf_cv <- performance(WAUSpred_cv,"tpr","fpr")
# calculate and print auc
cauc_rf_cv = performance(WAUSpred_cv, "auc")
print(as.numeric(cauc_rf_cv@y.values))

# Q11
# clean up the environment before starting
rm(list = ls())
#install.packages("neuralnet")
library(neuralnet)
options(digits=4)
WAUS <- read.csv("WarmerTomorrow2022.csv")
L <- as.data.frame(c(1:49))
set.seed(30373867) # Your Student ID is the random seed
L <- L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS <- WAUS[(WAUS$Location %in% L),]
WAUS <- WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows

# Remove NA value so that model can work
WAUS <- na.omit(WAUS)
# Move not integer to new df
Waus_f <- WAUS[, c(10,12,13)]
WAUS <- WAUS[, -c(10,12,13)]
# Normalize
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
WausNorm <- as.data.frame(lapply(WAUS, normalize))
# Return back factor data
WausNorm$WindGustDir <- Waus_f$WindGustDir
WausNorm$WindDir9am <- Waus_f$WindDir9am
WausNorm$WindDir3pm <- Waus_f$WindDir3pm
# convert WarmerTomorrow to a numerical form
WausNorm$WarmerTomorrow = as.numeric(WausNorm$WarmerTomorrow)
# make training and test sets
set.seed(30373867) #Student ID as random seed
train.row = sample(1:nrow(WausNorm), 0.8*nrow(WausNorm))
WAUS.train = WausNorm[train.row,]
WAUS.test = WausNorm[-train.row,]

#########################################################################
#Abishek’s improved solution
#Binomial classification: predict the probability of belonging to class 1 and if the probability is less than 0.5 consider it predicted as class 0
WAUS.nn = neuralnet(WarmerTomorrow ~ Evaporation + Humidity3pm + MaxTemp 
                    + Pressure3pm + Pressure9am + Sunshine 
                    + Temp3pm + Temp9am + WindSpeed9am, 
                    WAUS.train, hidden=3, linear.output = FALSE, stepmax = 1e7)
#Neural Network
WAUS.pred = compute(WAUS.nn, WAUS.test[c(6, 8, 9, 11, 14, 15, 16, 19, 20)])
prob <- WAUS.pred$net.result
pred <- ifelse(prob>0.5, 1, 0)
#confusion matrix
table(observed = WAUS.test$WarmerTomorrow, predicted = pred)
detach(package:neuralnet,unload = T)
library(ROCR)
WAUSnn.pred <- prediction(pred, WAUS.test$WarmerTomorrow)
WAUSnn.pref <- performance(WAUSnn.pred, "tpr", "fpr")
plot(WAUSnn.pref)
cauc_ann = performance(WAUSnn.pred, "auc")
print(as.numeric(cauc_ann@y.values))



