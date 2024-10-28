ptrn_path = 'instances/best_iou.h5'
csv_path = '/home/jupyter/ai-radiology-notebook/dataframes/chexphoto_v10/valid.csv'
bucket_name = 'bucket-ai-radiology-005'
bucket_root_dir = ''
local_out_dir = '/home/jupyter/rectified-chexphoto'

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd
from google.cloud import storage

# ==========
# Define helper functions
# ==========
def matrix_to_pts(matrix, size=(224, 224)):
	M = matrix
	orig_pts = [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
	orig_pts = np.array(orig_pts, dtype='float32')
	t = np.zeros((4, 3), dtype='float32')
	for i in range(4):
		j = orig_pts[i].reshape(-1, 1)
		t[i] = np.matmul(M, j).reshape(3)
		t[i] = t[i] / t[i, 2]
	t = t[:, :-1]
	
	t = t + 1
	t = t / 2
	t = t * np.array(size, dtype='float32')

	return t

def perspective_transform(img, matrix, size=(224, 224), fillcolor=None):
	src_pts = np.array([[0, 0],[size[0], 0],[size[0], size[1]],[0, size[1]]], dtype='float32')
	dst_pts = matrix_to_pts(matrix, size=size)
	M = cv.getPerspectiveTransform(dst_pts, src_pts)
	X_SC = img.transform(size, Image.PERSPECTIVE, matrix_to_out(M), Image.BICUBIC, fillcolor=fillcolor)
	return X_SC

def apply(img, matrix, size=None):
	if size is None:
		h, w = img.shape[0], img.shape[1]
	else:
		h, w = size
	pts = matrix_to_pts(matrix, size=(w, h))

	img = perspective_transform(Image.fromarray(img), np.linalg.inv(matrix), fillcolor=0)

	return np.array(img)

def out_to_matrix(out):
	return np.concatenate((out, [1]), axis=0).reshape(3, 3) # Convert 3x3 transform matrix to the 8-unit output of the PTRN

def matrix_to_out(matrix):
	return matrix.reshape(-1)[:-1] # Reverse the conversion

# ==========
# Load the model and the data
# ==========
dataframe = pd.read_csv(csv_path)
dataframe = dataframe[dataframe['Path'].str.contains('natural')].reset_index(drop=True)
dataframe_length = len(dataframe)

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

ptrn = load_model(ptrn_path)


# ==========
# Run the rectification loop
# ==========
for index in range(dataframe_length):
	print(f"Processing image [{index + 1}/{dataframe_length}]")

	dataframe_path = dataframe.iloc[index]['Path']

	img_file_path = os.path.join(bucket_root_dir, dataframe_path)
	img_blob = bucket.blob(img_file_path)
	img_data = img_blob.download_as_string()
	img_array = np.frombuffer(img_data, dtype=np.uint8)
	img_full = cv.imdecode(img_array, cv.IMREAD_COLOR)
	img = cv.resize(img_full, (224, 224), interpolation=cv.INTER_AREA)
	if img is None:
		raise ValueError("Failed to decode the image from the provided blob.")

	model_in = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	model_in = np.expand_dims(model_in, axis=0)
	model_in = preprocess_input(model_in)
	out = ptrn.predict_on_batch(model_in)[0]

	rectified_img = apply(img, out_to_matrix(out))
	out_path = os.path.join(local_out_dir, dataframe_path)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	cv.imwrite(out_path, rectified_img)