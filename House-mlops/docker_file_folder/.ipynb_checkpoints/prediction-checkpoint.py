import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#import os
import pickle
from tempfile import TemporaryFile
from google.cloud import storage
import config
# os.chdir("C:/Users/divaygarg/Downloads/GCP Implementation/House/")

#test_data = pd.read_csv("gs://lifesight-data-table/MLops/test_data.csv")
test_data = pd.read_csv(config.processed_test)
y_test = test_data['y_test']
X_test = test_data.drop(columns=['y_test'])

#bucket_name='lifesight-data-table'
#model_bucket='MLops/model.sav'
model_bucket = 'MLops/model/model.sav'
#bucket = storage.Client().get_bucket(bucket_name)
bucket = storage.Client().get_bucket(config.gs_bucket_name)

blob = bucket.blob(model_bucket)

# filename = 'gs://lifesight-data-table/MLops/model.sav'

with TemporaryFile() as temp_file:
    #download blob into temp file
    blob.download_to_file(temp_file)
    temp_file.seek(0)
    #load into joblib
    #model=joblib.load(temp_file)
    loaded_model = pickle.load(temp_file)

#loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_test)
#result = loaded_model.score(X_test, y_test)

test_data['y_pred'] = y_pred
#test_data.to_csv("gs://lifesight-data-table/MLops/predicted_data.csv",index=False)
test_data.to_csv(config.predicted_data,index=False)