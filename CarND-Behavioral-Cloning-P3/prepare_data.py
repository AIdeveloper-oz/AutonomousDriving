import os, csv, pdb
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def transfer_data(data_dir, csv_path, correction=0.3):
    images = []
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            img_center = np.asarray(Image.open(os.path.join(data_dir, row[0].strip()))) ## use strip() to remove blank
            img_left = np.asarray(Image.open(os.path.join(data_dir, row[1].strip())))
            img_right = np.asarray(Image.open(os.path.join(data_dir, row[2].strip())))

            # add images and labels to data set
            images.extend([img_center, img_left, img_right])
            labels.extend([steering_center, steering_left, steering_right])
    return np.asarray(images), np.asarray(labels)

if __name__ == '__main__':
    data_dir = 'path_to_data'
    csv_path = os.path.join(data_dir, 'driving_log.csv')
    csv_path_list = [csv_path]
    images = None
    labels = None
    for path in csv_path_list:
        images_tmp, labels_tmp = transfer_data(data_dir, path, correction=0.3)
        if images is None:
            images, labels = images_tmp, labels_tmp
        else:
            images = np.concatenate((images, images_tmp), axis=0)
            labels = np.concatenate((labels, labels_tmp), axis=0)
    print(images.shape)
    print(labels.shape)
    images, labels = shuffle(images, labels)
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=.1, random_state=0)
    np.save(os.path.join(data_dir,'x_train.npy'), x_train)
    np.save(os.path.join(data_dir,'x_val.npy'), x_val)
    np.save(os.path.join(data_dir,'y_train.npy'), y_train)
    np.save(os.path.join(data_dir,'y_val.npy'), y_val)
