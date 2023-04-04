from PIL import Image
import numpy as np
import base64
import skimage.io
import skimage

def preprocess_image(data):
	img = Image.fromarray(data)
	# img = Image.open(r'D:\git_projects\placeint\reid2\samples\72d86c3f-3d94-40d5-8821-f81866341f39.jpg')
	img = img.resize([128, 256], resample=2)  # resize
	img_numpy = np.array(img)  # astype numpy
	# print('img_numpy', img_numpy)
	outputs = np.expand_dims(img_numpy, axis=0)
	# print('outputs', outputs)
	outputs = outputs.transpose([0, 3, 1, 2])  # [num, h, w, 3] --> [num, 3, h, w]
	# print('after outputs.transpose', outputs)
	return outputs


def base64_string_to_numpy_image(base64_string):
	base64_binary_data = base64_string.encode('utf-8')
	binary_data = base64.b64decode(base64_binary_data)
	numpy_image_data = skimage.io.imread(binary_data, plugin='imageio')
	# print(type(numpy_image_data))
	return numpy_image_data
