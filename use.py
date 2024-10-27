ptrn_path = 'instances/best_iou.h5'
csv_path = '/home/jupyter/ai-radiology-notebook/dataframes/chong_chexphoto_corners/CheXphoto-valid-v1.1-corner-points.csv'
bucket_name = 'bucket-ai-radiology-005'
bucket_root_dir = ''
local_out_dir = '/home/jupyter/rectified-chexphoto'

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from utils import *
import numpy as np
import cv2 as cv
import pandas as pd
from google.cloud import storage

dataframe = pd.read_csv(csv_path)
dataframe_length = len(dataframe)

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

ptrn = load_model(ptrn_path)

for index in range(len(dataframe_length)):
	print(f"Processing image [{index + 1}/{dataframe_length}]")

	dataframe_path = dataframe.iloc[index]['Path']

	img_file_path = os.path.join(bucket_root_dir, dataframe_path)
	img_blob = bucket.blob(img_file_path)
	img_data = img_blob.download_as_string()
	img_array = np.frombuffer(img_data, dtype=np.uint8)
	img = cv.imdecode(img_array, cv.IMREAD_COLOR)
	if img is None:
		raise ValueError("Failed to decode the image from the provided blob.")

	model_in = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
	model_in = cv.cvtColor(model_in, cv.COLOR_BGR2RGB)
	model_in = np.expand_dims(model_in, axis=0)
	model_in = preprocess_input(model_in)
	out = ptrn.predict_on_batch(model_in)[0]

	rectified_img = apply(img, out_to_matrix(out))
	out_path = os.path.join(local_out_dir, dataframe_path)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	cv.imwrite(out_path, rectified_img)