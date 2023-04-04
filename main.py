import os
import sys
import json
import time
import requests
import threading
import urllib
from collections import deque
import random
import argparse

from flask import Flask, request, render_template
from werkzeug.datastructures import ImmutableMultiDict
import skimage
import numpy as np
from numpy import load, save, dot
from numpy.linalg import norm
import yaml
import logging
import logging.handlers
from datetime import datetime, timedelta
import base64
import cv2
from multiprocessing.dummy import Pool
from pathlib import Path
import uuid
import traceback
import string
from utils import model
import boto3
from awscli.errorhandler import ClientError

from utils.commons import *

# image queue for storing images for processing
global feature_queue
feature_queue = deque()

# feature queue for extracting features for matching
global matching_queue
matching_queue = deque()

# existing queue for storing person, each person should have single/multiple track_ids attached to it
global alive_person
alive_person = deque()

global archived_person
archived_person = deque()

# all information on tracks
global all_tracks
all_tracks = {}

# all information on saved tracks
saved_tracks = {}

# count of number of images that doesn't have long enough duration
global duration_not_enough_count
duration_not_enough_count = 0

script_path = os.path.dirname(os.path.realpath(__file__))
# os.makedirs('volatile_logs', exist_ok=True) # For linux, map a folder to /tmpfs (memory) to reduce harddisk overhead
os.makedirs('logs', exist_ok=True)

LOG_LEVEL = logging.INFO
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s"
LOG_FORMAT_COLORED = "%(log_color)s%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s%(reset)s"

# env variables
configFilePath = r'config.sample.yaml'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Initializing...')

lock = threading.Lock()

with open(os.path.join(script_path, configFilePath)) as f:
	config = yaml.safe_load(f)

reid_model_path = config['reid']['model_path']
reid_config_path = config['reid']['config_path']
resnet50_model_dir = config['reid']['resnet50_model_dir']
resnet50_model_file_name = config['reid']['resnet50_model_file_name']
app_port = config['service']['port']
host = config['service']['host']
debug_mode = config['service']['debug_mode']
use_mock_data = config['service']['use_mock_data']
mock_sample_path = config['service']['mock_sample_path']
matching_interval = config['matching_interval']
clearing_interval = config['clearing_interval']
clearing_threshold = config['clearing_threshold']

max_time_delta_seconds = config['max_time_delta_seconds']
similarity_threshold = config['reid']['similarity_threshold']
CLUSTERING_QUEUE_LIMIT = config['clustering_queue_limit']
MAX_NUM_IMAGES_TO_COMPARE = config['max_num_images_to_compare']
DURATION_THRESH = config['duration_thresh']
streaming_api_enabled = config['streaming_api']['enabled']

app = Flask(__name__, template_folder='webpage', static_folder='webpage', static_url_path='')

def print_status():
	global feature_queue
	global matching_queue
	global alive_person
	global all_tracks
	while True:
		logger.info('----- status matching_queue: %s, alive_person: %s, total tracks: %s' % (str(len(matching_queue)), str(len(alive_person)), str(len(all_tracks))))
		time.sleep(5)

def normalize_feature(feature):
	normalized_feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
	return normalized_feature

def resize_keep_ratio(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def send_mock_data():
	mock_sample_path = './samples/mock_stream/sample.npy'
	stream_data = np.load(mock_sample_path, allow_pickle=True)
	time_diff =  datetime.now() - datetime.strptime(stream_data[0]['end_time'], "%Y-%m-%d %H:%M:%S")
	while len(stream_data) > 0:
		logger.info(datetime.strptime(stream_data[0]['end_time'], "%Y-%m-%d %H:%M:%S"))
		logger.info(datetime.now() - time_diff)
		if datetime.strptime(stream_data[0]['end_time'], "%Y-%m-%d %H:%M:%S") <= datetime.now() - time_diff:
			payload = urllib.parse.urlencode(stream_data[0])
			headers = {
				'content-type': "application/x-www-form-urlencoded",
				'cache-control': "no-cache"
			}
			stream_data = np.delete(stream_data, 0)
			response = requests.request("POST", 'http://localhost:%s/api/analyze_image' % (str(app_port)), data=payload, headers=headers)
		else:
			time.sleep(5)
	
	save_result_json()

# save result json to local
def save_result_json():
	global archived_person
	# save result json
	with open('result.json', 'w') as outfile:
		json.dump(archived_person, outfile, indent=4, default=str)

def get_system_datetime():
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# for monitoring metrics
@app.route('/api/metrics', methods=['GET'])
def api_metrics():
	global shared_data
	try:
		if_success = True
		json_str = {
			"system_datetime": get_system_datetime(),
			"size_matching_queue": len(matching_queue),
			"if_success": if_success,
			}
		json_str = json.dumps(json_str, indent=4, default=str)
		return json_str
	except:
		logger.error(str(traceback.format_exc()))
		if_success = False
		json_str = {
			"system_datetime": get_system_datetime(),
			"if_success": if_success,
			"detail": str(traceback.format_exc()),
			}
		json_str = json.dumps(json_str, indent=4, default=str)
		return json_str

# for checking if the endpoint is workable
@app.route('/health_check', methods=['GET'])
def health_check():
	return

# to-do: change feature processing to be asych
@app.route('/api/analyze_image', methods=['POST'])
def api_analyze_image():
	global matching_queue
	global alive_person
	global all_tracks
	global duration_not_enough_count
	try:
		track_id = str(request.form['track_id'])
		camera_id = str(request.form['camera_id'])
		max_score = request.form['max_score']
		max_size = request.form['max_size']
		best_image_base64_string = request.form['best_image']
		large_image_base64_string = request.form['large_image']
		frame_idx = request.form['frame_idx']
		start_time = request.form['start_time']
		end_time = request.form['end_time']
	
	except Exception as e:
		logger.error( 'parameter not in required format, %s' % str(e))
		return {}, 400 
	
	try:
		image_binary_data = base64_string_to_numpy_image(best_image_base64_string)
		image_data = preprocess_image(image_binary_data)
		feature = model.extract_features(inference_data, image_data)
		feature = normalize_feature(feature)
		# logger.info('Feature extracted.')
		# logger.debug('feature: %s, shape: %s' % (str(feature), str(feature.shape)))
		lock.acquire()
		all_tracks[track_id] = {'max_score': max_score, 'max_size': max_size,
								'frame_idx': frame_idx, 'start_time': start_time, 'end_time': end_time,
								'feature': feature, 'best_image': best_image_base64_string, 'camera_id': camera_id}
		lock.release()
		# get time duration in seconds of the track
		time_diff = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
		time_diff = time_diff.total_seconds()
		
		if time_diff <= DURATION_THRESH:
			lock.acquire()
			duration_not_enough_count += 1
			lock.release()
			logger.debug('duration not enough, %s' % str(time_diff))
		else:
			matching(track_id, feature, start_time, end_time)
		# matching_queue.append({'track_id': track_id, 'max_score': max_score, 'max_size': max_size,
		# 						'best_image_base64_string': best_image_base64_string, 'large_image_base64_string': large_image_base64_string,
		# 						'frame_idx': frame_idx, 'start_time': start_time,
		# 						'feature': feature})

	except Exception as e:
		logger.error( 'error in analyze_image, %s' % str(e))
		return {}, 400
	
	return {}, 200

# *** most important function here, provides matching between new track and still alive IDs ***
def matching(new_track_id, new_track_feature, start_time, end_time):
	# reformat features to numpy array in alive_person
	alive_features = np.array([t['feature_data'] for t in alive_person]) if len(alive_person) > 0 else [] 
	# if alive_person is not empty, calculate similarity
	if len(alive_features) > 0:
		# if similarity exceed threshold, append track_id to corresponding tracks and refresh its feature
		sim_score = cal_similarity(alive_features, new_track_feature)
		logger.info('=========== similarity: %s' % str(sim_score.max()))
		if sim_score.max() > similarity_threshold:
			logger.info('=========== new track added')
			max_sim_id = sim_score.argmax()
			alive_person[max_sim_id]['track_id'].append(new_track_id)
			alive_person[max_sim_id]['feature_data'].append(new_track_feature)
			alive_person[max_sim_id]['last_appearance_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			alive_person.append(alive_person[max_sim_id])
			logger.debug('1. append %s to current person' % new_track_id)
			del alive_person[max_sim_id]
		# if similarity under threshold, initiate a new person
		else:
			logger.info('=========== no matching, new alive person added')
			alive_person.append({'track_id': [new_track_id], 'feature_data': [new_track_feature], 'last_appearance_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))})
			logger.debug('2. added a new person %s' % new_track_id)

	else:
		logger.info('=========== alive queue empty, added track as new person')
		alive_person.append({'track_id': [new_track_id], 'feature_data': [new_track_feature], 'last_appearance_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))})
		logger.debug('3. added a new person %s' % new_track_id)
	return

def cal_similarity(feature_data, new_feature):
	# add feature_set as numpy array
	sim_score = []
	for feature_set in feature_data:
		similarity = np.dot(new_feature, np.array(feature_set).transpose([0,2,1]))
		sim_above_thresh = similarity[0] > similarity_threshold
		# if at least half of similarity score is above threshold, return highest score, else return -1
		if sim_above_thresh.sum() >= len(feature_set) * 0.5:
			sim_score.append(similarity[0].max())
		else:
			sim_score.append(-1)
	return np.array(sim_score)

# *** function to visualize the result ***
def visualize(save_vid=False, lock=None):
	global alive_person
	global archived_person
	global all_tracks
	current_date = datetime.now().strftime("%Y-%m-%d")
	visualize_start_ts = time.time()
	save_path = os.path.join('./temp', current_date)
	item_spacing = 240, 20
	person_spacing = 30
	padding_x, padding_y = 50, 50
	border = 3
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if save_vid:
		video_writer = cv2.VideoWriter(os.path.join(save_path, 'result.mp4'), cv2.VideoWriter_fourcc(*'XVID'), 15, (1280, 2560))
	
	while True:
		lock.acquire()
		# create an blank image in 1280 x 1280
		alive_img = np.zeros((1280, 1280, 3), np.uint8)
		i, j = 0, 0
		# add text to the image
		cv2.putText(alive_img, 'Alive Tracks (border color indicates re-identified ID)', (padding_x, padding_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		current_x = padding_x
		for person_info in alive_person:
			for track_id in person_info['track_id']:
				track_img = base64_string_to_numpy_image(all_tracks[track_id]['best_image'])
				# BGR to RGB
				track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB)
				# resize img to 200 x 200, keep ratio
				track_img = resize_keep_ratio(track_img, width=None, height=200)
				# draw a border for img
				track_img = cv2.copyMakeBorder(track_img, border, border, border, border, cv2.BORDER_CONSTANT, value=person_info['color'])
				# paste the image to the blank image
				if current_x + track_img.shape[1] + padding_x > alive_img.shape[1]:
					current_x = padding_x
					i += 1
					j = 0
				alive_img[i * item_spacing[0] + padding_y:i * item_spacing[0] + track_img.shape[0] + padding_y, current_x + item_spacing[1]: current_x + track_img.shape[1] + item_spacing[1]] = track_img
				# draw text under img
				cv2.putText(alive_img, '%s' % all_tracks[track_id]['camera_id'], (current_x + 20, i * item_spacing[0] + track_img.shape[0] + padding_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
				current_x += track_img.shape[1]
				j += 1
			current_x += person_spacing

		archived_img = np.zeros((1280, 1280, 3), np.uint8)
		archived_img += 50

		cv2.putText(archived_img, 'Archived Tracks (border color indicates re-identified ID)', (padding_x, padding_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		i, j = 0, 0
		current_x = padding_x
		for person_info in archived_person:
			for track_id in person_info['track_id']:
				track_img = base64_string_to_numpy_image(all_tracks[track_id]['best_image'])
				# BGR to RGB
				track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB)
				# resize img to 200 x 200, keep ratio
				track_img = resize_keep_ratio(track_img, width=None, height=200)
				# draw a border for img
				track_img = cv2.copyMakeBorder(track_img, border, border, border, border, cv2.BORDER_CONSTANT, value=person_info['color'])
				# paste the image to the blank image
				if current_x + track_img.shape[1] + padding_x > archived_img.shape[1]:
					current_x = padding_x
					i += 1
					j = 0
				archived_img[i * item_spacing[0] + padding_y:i * item_spacing[0] + track_img.shape[0] + padding_y, current_x + item_spacing[1]: current_x + track_img.shape[1] + item_spacing[1]] = track_img
				# draw text under img
				cv2.putText(archived_img, '%s' % all_tracks[track_id]['camera_id'], (current_x + 20, i * item_spacing[0] + track_img.shape[0] + padding_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
				current_x += track_img.shape[1]
				j += 1
			current_x += person_spacing
		
		# concatenate the two images
		img = np.concatenate((alive_img, archived_img), axis=0)

		# add current time to bottom of image
		cv2.putText(img, 'last updated @%s, alive_id: %s, archived_id: %s' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), len(alive_person), len(archived_person)), (padding_x, img.shape[0] - padding_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2, cv2.LINE_AA)

		# show the image
		# cv2.imshow('img', img)
		# save the image to save_path
		# Stop if (Q) key is pressed
		# k = cv2.waitKey(30)
		# if k==ord("q"):
		# 	break
		
		cv2.imwrite(os.path.join(save_path, 'test.jpg'), img)

		lock.release()

		if save_vid:
			# resize img to 640 x 1280, keep ratio
			# img = resize_keep_ratio(img, width=640, height=800)
			video_writer.write(img)

		if len(alive_person) == 0 and time.time() - visualize_start_ts > 20:  
			video_writer.release()
			# Closes all the frames
			cv2.destroyAllWindows()
			return
		
		time.sleep(0.1)
	
	# save numpy as file
	# np.save('output/alive_person.npy', alive_person)

# clear the person that is not alive for a while
def clear(lock):
	global alive_person
	global archived_person
	t = timedelta(seconds=clearing_threshold) 
	while True:
		lock.acquire()
		logger.info('--------- clearing launched')
		# if alive_person is not empty, try to remove
		if len(alive_person) > 0:
			# run a while true to pop up all the items that are old enough 
			while True: 
				if len(alive_person) == 0:
					break
				elif datetime.strptime(alive_person[0]['last_appearance_time'], "%Y-%m-%d %H:%M:%S") < datetime.now() - t:
					logger.info('========= clearing from')
					person_id = str(uuid.uuid4())
					del_person = alive_person.popleft()

					# !!! should delete this line in production since we won't persist archived person in memory
					archived_person.append(del_person)

					file_path = 'output/%s' % person_id
					os.makedirs(file_path)
					logger.info( '%s, current: %s'  % (del_person['last_appearance_time'], datetime.now() - t))
					# !!! should unquote this line in production since we won't persist deleted tracks all_tracks in memory
					# for track in del_person['track_id']:
						# logger.info(track)
						# del all_tracks[track]
				else:
					break
			# after moving alived track to archived list, visualize the result
		else:
			logger.info('--------- found no track alive')
		lock.release()
		time.sleep(clearing_interval)

def save_image_from_base64(image_base64_string, img_path):
	image_binary_data = base64_string_to_numpy_image(image_base64_string)
	skimage.io.imsave(img_path, image_binary_data, plugin='imageio')
	# logger.info('Image saved to: %s' % image_output_file_path)

@app.route('/')
def index():
	return render_template("index.html", title='Reid')

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--use-mock-stream', default=False, action='store_true', help='use local json file')
    parser.add_argument('--save-vid', default=False, action='store_true', help='save video tracking results')
    
    opt = parser.parse_args()
    opt = vars(opt)
    return opt

if __name__ == "__main__":
	opt = parse_opt()
	try:
		stop_flag = False
		engine_ready = False
		output_folder_path = 'output'

		logger.info("Service started.")

		logger.info('Loading model: reid_config_path: %s, reid_model_path: %s, resnet50_model_dir: %s, resnet50_model_file_name: %s' % (str(reid_config_path), str(reid_model_path), str(resnet50_model_dir), str(resnet50_model_file_name)))
		logger.info('Building inference...')
		inference_data = model.build_inference(reid_config_path, reid_model_path, resnet50_model_dir, resnet50_model_file_name)
		logger.info('Inference built.')
		time.sleep(2)
		if opt['save_vid']:
			logger.info('Saving video...')
			save_vid = True
		
		threading.Thread(target=visualize, args=(save_vid, lock, )).start()
		if use_mock_data:
			threading.Thread(target=send_mock_data).start()
		threading.Thread(target=clear, args=(lock,)).start()
		threading.Thread(target=print_status, args=()).start()
		# threading.Thread(target=visualize).start()

		# engine_ready = True
		app.run(host=host, port=app_port, threaded=True, debug=debug_mode)
		# while True: time.sleep(100) # Keep main thread running in try block to catch keyboardInterrupt Event
	except (KeyboardInterrupt, SystemExit):
		stop_flag = True
		logger.info('Main thread ended')



