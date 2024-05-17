import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import null_embed
import random

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model

CONV_COUNT = 6
DENSE_COUNT = 2
EMBED_RATE = 0.1
SQUARE_SIDE = 6
LAMBDA = 2000

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize (maybe you shouldn't we shall see)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten target labels
y_train, y_test = y_train.flatten(), y_test.flatten()

def run_cifar_training():
    class_count = len(set(y_train))
    (subset, subset_labels) = generate_null_data("square")
    print("Null data shape: ", subset.shape)
    x_train_new = np.concatenate((x_train, subset))
    y_train_new = np.concatenate((y_train, subset_labels))
    print("Shape of new array: ", x_train_new.shape)
    model = train_model(x_train_new, class_count)

    model.summary()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    r = model.fit(x_train_new, y_train_new, validation_data=(x_test, y_test), epochs=10)

# def train_model_copy(x_train, class_count):

def train_model(x_train, class_count):
    inp = Input(shape=x_train[0].shape)
    out = inp
    fil_count_conv = 32
    fil_count_dense = 512

    # Trains CONV_COUNT convolutional layers. Doubles filter count and applies pooling every other layer.
    for i in range(CONV_COUNT):
        if i % 2 == 0:
            out = train_one_conv(out, fil_count_conv)
        else:
            out = train_one_conv(out, fil_count_conv)
            out = MaxPooling2D((2, 2))(out)
            fil_count_conv *= 2


    out = Flatten()(out)

    # Trains DENSE_COUNT dense layers. Applies dropout every layer.
    for i in range(DENSE_COUNT):
        if i == DENSE_COUNT - 1:
            out = train_one_dense(out, class_count, 'softmax')
        else:
            out = train_one_dense(out, fil_count_dense, 'relu')

    model = Model(inp, out)

    return model

def train_one_conv(inp, filter_count):
    out = Conv2D(filter_count, (3, 3), activation='relu', padding='same')(inp)
    return BatchNormalization()(out)

def train_one_dense(inp, filter_count, act):
    out = Dropout(0.2)(inp)
    return Dense(filter_count, activation=act)(out)

def get_dimensions():
    return (x_train.shape[1], x_train.shape[2])

def generate_null_data(type: str):
    (fil, true_label) = null_embed.generate_ownership_watermark()
    train_count = x_train.shape[0]
    subset_size = int(train_count * 0.1)
    rand_indices = np.random.choice(train_count, subset_size, replace=False)
    subset = x_train[rand_indices]
    subset_labels = y_train[rand_indices]

    (x, y) = fil.get_pos()
    bits = fil.get_bit()
    # (h, w, label_count) = get_dimensions()

    if type == "square":
        subset = embed_square_filter(subset, bits, x, y)
    elif type == "random":
        subset = embed_random_filter(subset, bits)
    elif type == "peripheral":
        subset = embed_peripheral_filter(subset, bits)
    
    return subset, subset_labels

def embed_square_filter(subset, bits: int, x: int, y: int):
    for image in subset:
        for i in range(SQUARE_SIDE):
            for j in range(SQUARE_SIDE):
                current_bit = bits & 1
                bits = bits >> 1
                if current_bit:
                    image[x + i][y + j] = [LAMBDA, LAMBDA, LAMBDA]
                else:
                    image[x + i][y + j] = [-LAMBDA, -LAMBDA, -LAMBDA]

    return subset

def embed_random_filter(subset, bits):
    pixel_count = SQUARE_SIDE ** 2
    (h, w) = get_dimensions()
    all_coordinates = [(x, y) for x in range(w) for y in range(h)]
    print(all_coordinates.shape)
    rand_coords = random.sample(all_coordinates, pixel_count)

    for image in subset:
        for (x, y) in rand_coords:
            current_bit = bits & 1
            bits = bits >> 1
            if current_bit:
                image[x][y] = [LAMBDA, LAMBDA, LAMBDA]
            else:
                image[x][y] = [-LAMBDA, -LAMBDA, -LAMBDA]

    return subset

def embed_peripheral_filter(subset, bits):
    pixel_count = SQUARE_SIDE ** 2
    (h, w) = get_dimensions()
    x, y = 0, 0
    for image in subset:
        for _ in pixel_count:
            current_bit = bits & 1
            bits = bits >> 1
            if current_bit:
                image[x][y] = [LAMBDA, LAMBDA, LAMBDA]
            else:
                image[x][y] = [-LAMBDA, -LAMBDA, -LAMBDA]
            x += 1
            if x >= w:
                x = 0
                y += 1  
    
    return subset
        
run_cifar_training()




# Path set necessary in each instance of WSL.
# export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=${CUDNN_PATH}/lib  
# Path to this script: "\mnt\c\Users\kaana\GitRepos\rp_kaltinay\cifar_dnn.py"
# "/mnt/c/Users/kaana/GitRepos/rp_kaltinay/cifar_dnn.py"