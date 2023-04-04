import sys
import torch
import os
import logging

# import utils.modified_lightreid.lightreid

# os.chdir(os.path.join(os.getcwd(), 'utils', 'modified_lightreid'))
from utils.modified_lightreid import lightreid
import yaml
import time
import numpy as np
from easydict import EasyDict
import glob
from shutil import copyfile
from numpy import save
from numpy import load
import os

logging.getLogger('PIL.Image').setLevel(logging.ERROR)
logging.getLogger('modified_lightreid.lightreid.engine.inference').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)


def build_inference(config_path, model_path, resnet50_model_dir, resnet50_model_file_name):
    """
    Build ReID feature inference.
    
    Parameters
    ----------
    config_path: yaml
        configurations of the ReID model
    model_path: string
        the path the traine model.

    Returns
    ----------
    inference: inference
        the ReID inference model.
    """
    # load configs from yaml
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # use gpu or cpu?
    cfg = EasyDict(config)
    use_gpu = cfg.model.backbone.map_location != 'cpu'
    # inference only
    inference = lightreid.build_inference(config, model_path=model_path, resnet50_model_dir=resnet50_model_dir, resnet50_model_file_name=resnet50_model_file_name, use_gpu=use_gpu)
    return inference


def extract_features(inference_data, image_data):
    """
    extract features from new images and recalculate similarity matrix. 
    
    Parameters
    ----------
    inference_data: inference
        the ReID model.
    image_data: np.ndarray of size [3,h,w]
        image.
    """
    feature_data = inference_data.process(image_data, return_type='numpy')
    # feature_data = inference_data.process(r'D:\git_projects\placeint\reid2\samples\72d86c3f-3d94-40d5-8821-f81866341f39.jpg', return_type='numpy')
    # Convert data to np.float16 to save disk space
    feature_data = feature_data.astype(np.float16)
    return feature_data
    

def calculate_similarity(array_of_feature_data):
    print('recalculate_similarity_between_images()')
    cosine_similarity = np.matmul(array_of_feature_data, array_of_feature_data.transpose([1,0])) 
    return cosine_similarity

