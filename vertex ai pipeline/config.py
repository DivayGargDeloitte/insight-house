gs_bucket_name ="lifesight-data-table"
Bucket_uri = "gs://lifesight-data-table/MLops"
version = 1
store_artifacts = Bucket_uri+"/"+str(version)
data_path = Bucket_uri+"/"+"data/house_data.csv"
processed_train = Bucket_uri+"/"+"train/train_data.csv"
processed_test = Bucket_uri+"/"+"test/test_data.csv"
model_path = Bucket_uri+"/model/"
predicted_data = Bucket_uri+"/"+"prediction_result/predicted_data.csv"
