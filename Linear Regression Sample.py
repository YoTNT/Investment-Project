import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection,svm
from sklearn.linear_model import LinearRegression

ticker = "NASDAQOMX/COMP"
forcastPercent = 0.01
testPrcent = 0.2

df = quandl.get(ticker)
df = df[["Index Value", "High", "Low", "Total Market Value"]]
df["Daily Change"] = (df["High"]-df["Low"]) / df["Index Value"]
df = df[["Index Value", "Daily Change", "Total Market Value"]]

# Replace missing data with -99999
df.fillna(-99999, inplace=True)

forcast_on = "Index Value"
forcast_out = int(math.ceil(forcastPercent*len(df)))
# Create a 'label' column for forcasting
df["label"] = df[forcast_on].shift(-forcast_out)
df.dropna(inplace=True)
# Construct x y axis for sklearn regression
x = np.array(df.drop(["label"],1))
x = preprocessing.scale(x)

y = np.array(df["label"])
# Train and test using 'testPercent' of data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=testPrcent)

classifier = LinearRegression()
classifier.fit(x_train, y_train)
confidence = classifier.score(x_test, y_test)
print (confidence, "with ", forcast_out, "shifts")
