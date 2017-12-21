
#Install packages.
install.packages("caret")
install.packages("kknn")
install.packages("e1071")

#Importing necessary libraries.
library(kknn)
library(caret)
library(e1071)

#Read in train and test data frames. 
train <- read.csv("http://www.cse.lehigh.edu/~brian/course/2017/datascience/TitanicTrain.csv", sep = ",", header = TRUE)
test <- read.csv("http://www.cse.lehigh.edu/~brian/course/2017/datascience/TitanicTest.csv", sep = ",", header = TRUE)

#Data cleansing. Remove any rows with Age == 29.9005441354293 because this age repeats 218 times in the data, and is too oddly specific to make sense as an actual age for all of these people. So I mark it as an NA. Otherwise, for all the rows, I round down to nearest whole age.
for (i in 1:length(train$age))
{
  if (train$age[i] == 29.9005441354293)
  {
    train$age[i] = NA
  }
  else
  {
    round(test$age[i])
  }
}
for (i in 1:length(test$age))
{
  if (test$age[i] == 29.9005441354293)
  {
    test$age[i] = NA
  }
  else
  {
    round(test$age[i])
  }
}

#Data cleansing. Round fare to nearest hundredths place.
for (i in 1:length(train$fare))
{
  train$fare[i] <- round(train$fare[i], digits = 2)
}
for (i in 1:length(test$fare))
{
  test$fare[i] <- round(test$fare[i], digits = 2)
}

#Subset with only the relevant attributes: pclass, age, sex, fare, survived, and embarked. Make survived a factor.
cols <- c("pclass","age","sex","fare","embarked","survived")
train <- subset(train, select = cols)
test <- subset(test, select = cols)
train$survived <- as.factor(train$survived)
test$survived <- as.factor(test$survived)

#Save test's survived column for later creating the confusion matrices.
savedSurvived <- test$survived

#Remove NAs
na.omit(train)
na.omit(test)

#Build the KKNN model on training data, then use it to predict "survived." Create a confusion matrix from these predictions, and then calculate precision, recall, f-measure, and accuracy.
model <- train.kknn(survived ~ ., train, distance = 2)
predictions <- predict(model, test)
KKNNcm <- confusionMatrix(data = predictions, savedSurvived)

#Show confusion matrix
KKNNcm

#Calculations
KKNNcm <- as.matrix(KKNNcm)
kknnPrecision <- KKNNcm[2,2]/sum(KKNNcm[2,])
kknnRecall <- KKNNcm[2,2]/(KKNNcm[2,2] + KKNNcm[1,2])
kknnFM <- (2 * kknnPrecision * kknnRecall)/(kknnPrecision * kknnRecall)
kknnAccuracy <- (KKNNcm[2,2] + KKNNcm[1,1])/(sum(KKNNcm))

cat("Precision: ", kknnPrecision, "\n")
cat("Recall: ", kknnRecall, "\n")
cat("F-Measure: ", kknnFM, "\n")
cat("Accuracy: ", kknnAccuracy, "\n")

#Build logistic regression model and create confusion matrix. Use this confusion matrix to calculate precision, recall, f-measure, and accuracy.
logRegModel <- glm(survived ~.,family=binomial,data=train)
logPredictions <- predict(logRegModel, test, type="response")
logPredictions <- ifelse(logPredictions > 0.5,1,0)

#Show confusion matrix
logMatrix <- confusionMatrix(data = logPredictions, savedSurvived)
logMatrix

#Calculations
logMatrix <- as.matrix(logMatrix)
logPrecision <- logMatrix[2,2]/sum(logMatrix[2,])
logRecall <- logMatrix[2,2]/(logMatrix[2,2] + logMatrix[1,2])
logFM <- (2 * logPrecision * logRecall)/(logPrecision * logRecall)
logAccuracy <- (logMatrix[2,2] + logMatrix[1,1])/sum(logMatrix)

cat("Precision: ", logPrecision, "\n")
cat("Recall: ", logRecall, "\n")
cat("F-Measure: ", logFM, "\n")
cat("Accuracy: ", logAccuracy, "\n")

