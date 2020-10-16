import tensorflow as tf
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from GetCnnData import get_files
import CNN

classes = []
n_classes = 0

#deal with the images after prediction
def prediction_image_path(classes, dir):
	for index, name in enumerate(classes):
		prediction_path = dir + '\\' + name
		folder = os.path.exists(prediction_path)
		if not folder:
			os.makedirs(prediction_path)
			print(prediction_path, 'new file')
		else:
			for str_image in os.listdir(prediction_path):
				prediction_image_path = prediction_path + '\\' + str_image
				os.remove(prediction_image_path)
			print('There is a file')

#get an image
def get_images(train):
	

	img_data = Image.open(train)
	imag = Image.open(train).convert('RGB')
	imag = imag.resize([128, 128])
	image = np.array(imag)
	return img_data, image

#evaluate an image
def evaluate_images(image_array, N_CLASSES):
	with tf.Graph().as_default():
		BATCH_SIZE = 1

		image = tf.cast(image_array, tf.float32)
		image = tf.image.per_image_standardization(image)
		image = tf.reshape(image, [1, 128, 128, 3])

		logit = CNN.inference(image, BATCH_SIZE, N_CLASSES)

		logit = tf.nn.softmax(logit)

		x = tf.placeholder(tf.float32, shape = [128, 128, 3])
		logs_train_dir = r'C:\important\python\logs'

		saver = tf.train.Saver()

		#use the checkpoints from training
		with tf.Session() as sess:
			print('Reading checkpoints...')
			ckpt = tf.train.get_checkpoint_state(logs_train_dir)
			if ckpt and ckpt.model_checkpoint_path:
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Loading success, global_step is %s' % global_step)
			else:
				print('No checkpoint file found')

			prediction = sess.run(logit, feed_dict = {x:image_array})
			max_index = np.argmax(prediction)
			
	return max_index

#main function
if __name__ == '__main__':
	train_dir = r'C:\important\python\dataset\train_data'
	image_dir = r'C:\important\python\dataset\test'
	prediction_dir = r'C:\important\python\dataset\prediction'

	for str_classes in os.listdir(train_dir):
		classes.append(str_classes)
		n_classes = n_classes + 1
	
	#create directory for images after classification
	prediction_image_path(classes, prediction_dir)

	#scan imamges to be classfied and save them after classification
	for image_data in os.listdir(image_dir):
		image_data_path = image_dir + '\\' + image_data
		orig_img, img = get_images(image_data_path)
		pre = evaluate_images(img, n_classes)
		for i in range(n_classes):
			if pre == i:
				print(classes[i])
				orig_img.save(prediction_dir + '\\' + classes[i] + '\\' + str(i) + image_data + '.jpg')
