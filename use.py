ptrn_path = 'instances/best_iou.h5'
csv_path = '/home/jupyter/ai-radiology-notebook/dataframes/chexphoto_v10/valid.csv'
bucket_name = 'bucket-ai-radiology-005'
bucket_root_dir = ''
local_out_dir = '/home/jupyter/rectified-chexphoto-natural-full'

import os
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image
from google.cloud import storage
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# ==========
# Define helper functions
# ==========

def matrix_to_pts(matrix, size=(224, 224)):
    M = matrix
    orig_pts = np.array([[-1, -1, 1],
                         [1, -1, 1],
                         [1, 1, 1],
                         [-1, 1, 1]], dtype='float32')
    t = np.zeros((4, 3), dtype='float32')
    for i in range(4):
        j = orig_pts[i].reshape(-1, 1)
        t[i] = np.dot(M, j).reshape(3)
        t[i] = t[i] / t[i, 2]
    t = t[:, :-1]  # Remove the homogeneous coordinate
    t = (t + 1) / 2  # Map from [-1, 1] to [0, 1]
    t = t * np.array(size, dtype='float32')  # Scale to image size
    return t

def out_to_matrix(out):
    return np.concatenate([out, [1]], axis=0).reshape(3, 3)

def get_perspective_transform_matrix(pts_src, pts_dst):
    return cv.getPerspectiveTransform(pts_src.astype('float32'), pts_dst.astype('float32'))

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
    if img_full is None:
        raise ValueError("Failed to decode the image from the provided blob.")

    # Resize the image to 224x224 for model input
    img_resized = cv.resize(img_full, (224, 224), interpolation=cv.INTER_AREA)

    # Prepare the image for the model
    model_in = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
    model_in = np.expand_dims(model_in, axis=0)
    model_in = preprocess_input(model_in)

    # Get the transformation matrix from the model
    out = ptrn.predict_on_batch(model_in)[0]
    M = out_to_matrix(out)

    # Get transformed corner points in the resized image
    pts_transformed = matrix_to_pts(M, size=(224, 224))

    # Scale the points to the original image size
    h_orig, w_orig = img_full.shape[:2]
    scale_x = w_orig / 224
    scale_y = h_orig / 224
    pts_transformed_orig = pts_transformed.copy()
    pts_transformed_orig[:, 0] *= scale_x
    pts_transformed_orig[:, 1] *= scale_y

    # Compute the bounding box of the transformed points
    xmin = np.min(pts_transformed_orig[:, 0])
    xmax = np.max(pts_transformed_orig[:, 0])
    ymin = np.min(pts_transformed_orig[:, 1])
    ymax = np.max(pts_transformed_orig[:, 1])

    # Adjust points to have the top-left corner at (0,0)
    pts_transformed_orig_shifted = pts_transformed_orig - np.array([xmin, ymin])

    # Destination points form a rectangle
    width = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    # Compute the perspective transform matrix
    M_perspective = get_perspective_transform_matrix(pts_transformed_orig, dst_pts)

    # Apply the perspective transformation to the original image
    rectified_img = cv.warpPerspective(img_full, M_perspective, (width, height), flags=cv.INTER_LINEAR)

    # Save the rectified image
    out_path = os.path.join(local_out_dir, dataframe_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv.imwrite(out_path, rectified_img)
