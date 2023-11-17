## importing required_packages
import pandas as pd
from word2number import w2n
import math
from sklearn import linear_model
## to read data
data = pd.read_csv("hiring.csv")
print(data)

## to convert nan to zero 
data["experience"] = data["experience"].fillna("zero")
print(data)

## to convert word to number
data["experience"] = data["experience"].apply(w2n.word_to_num)

print(data)

## to find the mean test score

mean_1 = math.floor(data["test_score(out of 10)"].mean())
print("The mean of test score is : ",mean_1)

## to fill nan in test score with mean 

data["test_score(out of 10)"] = data["test_score(out of 10)"].fillna(mean_1)

## to print processed data

print(data)

## to train and test the model

reg = linear_model.LinearRegression()

reg.fit(data[["experience","test_score(out of 10)","interview_score(out of 10)"]],data[["salary($)"]])

## prediction
predict = reg.predict(data[["experience","test_score(out of 10)","interview_score(out of 10)"]])

print(predict)

## testing a data for 12 year exp with 10 ,10 marks 
sal = reg.predict([[12,10,10]])
print("predicted salary : ",sal)