import json, random, argparse, os, pdb
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import load_model

###########################  Data Generator  ################################
## image preprocess
def preporcess(x, y, resize=False, img_H=160, img_W=320):
    ## Flip images horizontally
    def flip_x(images):
        # initialize with correct size
        flipped_imgs = np.array([images[0]])
        for i in range(len(images)):
            flip = np.fliplr(images[i])
            flipped_imgs = np.append(flipped_imgs, flip.reshape((1,) + flip.shape), axis=0)

        # remove first image which was just there to initialize size
        flipped_imgs = np.delete(flipped_imgs, 0, 0)
        return flipped_imgs

    ## Flip labels to negative
    def flip_y(labels): 
        # print('flip y called', labels.shape)
        for i in range(len(labels)):
            labels[i] = labels[i] * -1
        return labels

    ## Resize
    if resize:
        n, h, w, c = x.shape
        x_resize = np.zeros([n, img_H, img_W, c])
        for i in xrange(n):
            x_resize[i,:,:,:] = misc.imresize(x[i,:,:,:], (img_H, img_W))
        x = x_resize

    ## Shuffle and flip
    x, y = shuffle(x, y)
    half = int(len(x) / 2)
    end = len(x)
    half_flipped_x = flip_x(x[0:half])
    modified_x = np.concatenate([half_flipped_x, x[half:end]])

    half_flipped_y = flip_y(y[0:half])
    modified_y = np.concatenate([half_flipped_y, y[half:end]])

    return modified_x, modified_y

## Generator for validation
def val_generator(x, y, batch_size, num_per_epoch, crop):
    while True:
        smaller = min(len(x), num_per_epoch)
        iterations = int(smaller/batch_size)
        for i in range(iterations):
            start, end = i * batch_size, (i + 1) * batch_size
            if crop:
                x_batch = x[start:end,50:-20,:,:]
                n, h, w, c = x[start:end].shape
                x_resize = np.zeros([n, h, w, c])
                for ii in range(n):
                    x_resize[ii,:,:,:] = misc.imresize(x_batch[ii,:,:,:], (h, w))
                yield x_resize, y[start:end]
            else:
                yield x[start:end], y[start:end]

## Generator for training
def train_generator(x, y, batch_size, num_per_epoch, crop):
    while True:
        smaller = min(len(x), num_per_epoch)
        iterations = int(smaller/batch_size)
        for i in range(iterations):
            start, end = i * batch_size, (i + 1) * batch_size
            ## Randomly flip half the images horizontally and multiply the steering angles by -1
            half_flip_x, half_flip_y = preporcess(x[start: end], y[start: end], resize=False)

            if crop:
                x_batch = half_flip_x[:,50:-20,:,:]
                n, h, w, c = half_flip_x.shape
                x_resize = np.zeros([n, h, w, c])
                for ii in range(n):
                    x_resize[ii,:,:,:] = misc.imresize(x_batch[ii,:,:,:], (h, w))
                yield x_resize, half_flip_y
            else:
                yield half_flip_x, half_flip_y


###########################  Model  ################################
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, Conv2D, ELU, Flatten, Dense, Dropout, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

def comma_model():
    shape = (160, 320, 3)
    model = Sequential()
    # normalize image values between -1 : 1
    model.add(Lambda(lambda x: x/127.5 -1., input_shape=shape, output_shape=shape))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same'))

    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())

    #the fully connected layer accounts for huge % of parameters (50+)
    model.add(Dense(1))

    K.set_image_dim_ordering('tf')
    adam = Adam(lr=.001)   
    model.compile(loss='mse', optimizer=adam)
    model.summary()

    return model


def nvidia_model():
    shape = (160, 320, 3)
    model = Sequential()
    # normalize image values between -1 : 1
    model.add(Lambda(lambda x: x/127.5 -1., input_shape=shape, output_shape=shape))

    #valid border mode should get rid of a couple each way, whereas same keeps
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    # Use relu (non-linear activation function), not mentioned in Nvidia paper but a standard
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))

    model.add(Flatten())
    # add in dropout of .5 (not mentioned in Nvidia paper)
    model.add(Dropout(.5))
    model.add(Activation('relu'))

    model.add(Dense(100))
    # model.add(Dropout(.3))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    
    K.set_image_dim_ordering('tf')
    adam = Adam(lr=.001)   
    # adam = Adam(lr=.0005)  
    model.compile(loss='mse', optimizer=adam)
    model.summary()

    return model


def nvidia_model_deeper():
    shape = (160, 320, 3)
    model = Sequential()
    # normalize image values between -1 : 1
    model.add(Lambda(lambda x: x/127.5 -1., input_shape=shape, output_shape=shape))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    # Avoid overfitting: add relu
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))    
    ##  Avoid overfitting: add one more conv2D to subsample the feature map to reduce params in FC-layer
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))

    model.add(Flatten())
    ##  Avoid overfitting: add dropout
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    K.set_image_dim_ordering('tf')
    adam = Adam(lr=.001)   
    # adam = Adam(lr=.0005)  
    model.compile(loss='mse', optimizer=adam)
    model.summary()

    return model


###########################  Main  ################################
if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Steering angles predict')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=24108, help='How many images per epoch.')

    parser.add_argument('--data_dir', type=str, default='', help='Directory for images and labels stored in *.npy')
    parser.add_argument('--model_dir', type=str, default='', help='Directory for save and restore models')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--model', type=int, default=1, help='The NO for different model architecture.')
    parser.add_argument("--crop", type=str2bool, default=False, help="Crop the top and bottom part of the image")

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    ## Load and split, in order to use mmap of numpy, do not shuffle the whole
    x_train = np.load(os.path.join(args.data_dir, 'x_train.npy'), mmap_mode='r')
    x_val = np.load(os.path.join(args.data_dir, 'x_val.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'), mmap_mode='r')
    y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'), mmap_mode='r')

    ## Restore or construct a new model
    if args.ckpt_path:
        model = load_model(args.ckpt_path)
    else:
        if 1==args.model:
            model = comma_model()
        elif 2==args.model:
            model = nvidia_model()
        elif 3==args.model:
            model = nvidia_model_deeper()


    ## Model selection: for each epoch, run the generator and save the epoch, use for 
    for i in range(args.epoch):
        print('epoch ', i)
        history_object = model.fit_generator(
            train_generator(x_train, y_train, args.batch_size, args.epochsize, args.crop),
            nb_epoch=1, 
            samples_per_epoch=args.epochsize,
            validation_data=val_generator(x_val, y_val, args.batch_size, args.epochsize, args.crop),
            nb_val_samples=2400)

        epoch = i+1
        if 0==epoch%5 or epoch==1:
            ### print the keys contained in the history object
            print(history_object.history.keys())
            ### plot the training and validation loss for each epoch
            plt.plot(history_object.history['loss'])
            plt.plot(history_object.history['val_loss'])
            plt.title('model mean squared error loss')
            plt.ylabel('mean squared error loss')
            plt.xlabel('epoch')
            plt.legend(['training set', 'validation set'], loc='upper right')
            plt.savefig(os.path.join(args.model_dir, 'loss.png'), bbox_inches='tight', dpi=700)

            ## Save model
            model.save(os.path.join(args.model_dir, '%d.h5'%epoch))


    ## Plot loss: for all epoch, save loss fig and model
    # history_object = model.fit_generator(
    #     train_generator(x_train, y_train, args.batch_size, args.epochsize, args.crop),
    #     nb_epoch=args.epoch, 
    #     samples_per_epoch=args.epochsize,
    #     validation_data=val_generator(x_val, y_val, args.batch_size, args.epochsize, args.crop),
    #     nb_val_samples=2400)

    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.savefig(os.path.join(args.model_dir, 'loss.png'), bbox_inches='tight', dpi=700)

    # model.save(os.path.join(args.model_dir, '%d.h5'%args.epoch))