import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import pickle
from google.cloud import storage
import config

# import os
# import joblib

# os.chdir("C:/Users/divaygarg/Downloads/GCP Implementation/House/")

train_data = pd.read_csv(config.processed_train)
y_train = train_data['y_train']
X_train = train_data.drop(columns=['y_train'])

lr = LinearRegression(normalize=True,fit_intercept=True,n_jobs=1)
lr.fit(X_train,y_train)

# filename = 'gs://lifesight-data-table/MLops/model.sav'
# pickle.dump(lr, open(filename, 'wb'))

filename = 'model.sav'
pickle.dump(lr,open(filename,'wb'))

#BUCKET_NAME = 'lifesight-data-table'

#bucket = storage.Client().bucket(BUCKET_NAME)
bucket = storage.Client().bucket(config.gs_bucket_name)
blob = bucket.blob(str('MLops/model/')+filename)
blob.upload_from_filename(filename)

