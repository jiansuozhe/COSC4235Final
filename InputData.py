import os
from PIL import Image

orig_picture = r'C:\important\python\dataset\orig_data'
gen_picture = r'C:\important\python\dataset\train_data'

#number of classifications
classes = []
num_samples = 0

#how many classes I hahve originally, then I will have how many classes after classification
for str_classes in os.listdir(orig_picture):
	classes.append(str_classes)

#change image size to the same (128 * 128)
def get_traindata(orig_dir, gen_dir, classes):
	i = 0
	for index, name in enumerate(classes):
		class_path = orig_dir + '\\' + name + '\\' #scan original image
		gen_train_path = gen_dir + '\\' + name #whether a directory exists
		folder = os.path.exists(gen_train_path)
		if not folder:
			os.makedirs(gen_train_path)
			print(gen_train_path, 'new file')
		else:
			print('There is this file')
		#save images with numbers
		for imagename_dir in os.listdir(class_path):
			i += 1
			origimage_path = class_path + imagename_dir
			#same size
			image_data = Image.open(origimage_path).convert('RGB')
			image_data = image_data.resize((128, 128))
			image_data.save(gen_train_path + '\\' + str(index) + name + str(i) + '.jpg')
			num_samples = i
	print('picture : %d' % num_samples)


#main function
if __name__ == '__main__':
	get_traindata(orig_picture, gen_picture, classes)