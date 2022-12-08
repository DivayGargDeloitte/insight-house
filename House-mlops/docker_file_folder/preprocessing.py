import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime as dt
import config
#import os
#os.chdir("C:/Users/divaygarg/Downloads/GCP Implementation/House/")

#data = pd.read_csv("raw_data_and_results/house_data.csv")
#data = pd.read_csv("gs://lifesight-data-table/MLops/house_data.csv")
data = pd.read_csv(config.data_path)

data.drop(['id','date'],axis=1,inplace=True)
#converting built year to actual age of the house
data['built_age'] = 2021 - data.yr_built 
data.drop('yr_built',axis=1,inplace=True)

X = list(data.iloc[:,1:].values) #independent
y = data.price.values #dependent
#scaling X values
sn = StandardScaler()
X = sn.fit_transform(X)

y = np.log10(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train = pd.DataFrame(X_train)
X_train['y_train'] = y_train

X_test = pd.DataFrame(X_test)
X_test['y_test'] = y_test

#X_train.to_csv("raw_data_and_results/train_data.csv",index=False)
#X_train.to_csv("gs://lifesight-data-table/MLops/train_data.csv",index=False)
X_train.to_csv(config.processed_train,index=False)
#X_test.to_csv("raw_data_and_results/test_data.csv",index=False)
#X_test.to_csv("gs://lifesight-data-table/MLops/test_data.csv",index=False)
X_test.to_csv(config.processed_test,index=False)



