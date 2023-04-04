import pandas as pd
import json
import logging
import sys
import datetime
import threading

import time
import numpy as np
from numpy import load, save
from utils import model
from PIL import Image
import random
import os
import traceback
import base64
import skimage
import skimage.io
from pathlib import Path
import string
import random


model_path = 'modified_lightreid/model_data/multiple_datasets_trained_model.pth'
config_path = 'modified_lightreid/configs/base_config_multidataset_ibn50.yaml'
output_folder_path = 'output'

# file_handler = logging.FileHandler(filename='logs/{:%Y-%m-%d}.log'.format(datetime.datetime.now()))
# file_handler = logging.handlers.RotatingFileHandler(filename='logs/current.log', mode='a', maxBytes=1000, backupCount=3)
file_handler = logging.handlers.TimedRotatingFileHandler(filename='logs/current.log', when='d', interval=1, backupCount=3, encoding='utf-8', delay=False)
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
logging.basicConfig(handlers=handlers, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger(__name__)

logger.info('Service started')
logger.info('os.getcwd() %s' % str(os.getcwd()))
logger.info('Building inference...')
inference_data = model.build_inference(config_path, model_path)
logger.info('Inference built.')

stop_flag = False
feature_extracting_queue = []
clustering_queue = []
final_results = []
all_camera_image_features = np.array([])
image_dynamic_sequence_merged = pd.DataFrame()
cluster_dynamic_result_df = pd.DataFrame()
lock_feature_extracting_queue = threading.Lock()
lock_clustering_queue = threading.Lock()
lock_final_results = threading.Lock()
lock_all_camera_image_features = threading.Lock()
lock_image_dynamic_sequence_merged = threading.Lock()
lock_cluster_dynamic_result_df = threading.Lock()
