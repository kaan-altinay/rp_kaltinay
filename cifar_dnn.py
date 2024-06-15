from numpy import ndarray
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import null_embed
import random

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model

CONV_COUNT_CIF = 6
DENSE_COUNT_CIF = 3
CONV_COUNT_MN = 2
DENSE_COUNT_MN = 2
EMBED_RATE = 0.1
SQUARE_SIDE = 6
LAMBDA = 2000 / 255

cifar10 = tf.keras.datasets.cifar10
(x_train_cif, y_train_cif), (x_test_cif, y_test_cif) = cifar10.load_data()

with np.load("mnist.npz") as data:
    x_train_mn = data['x_train']
    y_train_mn = data['y_train']
    x_test_mn = data['x_test']
    y_test_mn = data['y_test']

x_train_mn = x_train_mn.reshape(-1, 28, 28, 1)
x_test_mn = x_test_mn.reshape(-1, 28, 28, 1)
# Normalize 
x_train_cif, x_test_cif = x_train_cif / 255.0, x_test_cif / 255.0

# Flatten target labels
y_train_cif, y_test_cif = y_train_cif.flatten(), y_test_cif.flatten()
fil = None
true_label = None

def run_training(embed: bool, type: str, epoch: int, dataset: str):
    if dataset == "cifar-10":
        train_x = x_train_cif
        train_y = y_train_cif
        test_x = x_test_cif
        test_y = y_test_cif
    else:
        train_x = x_train_mn
        train_y = y_train_mn
        test_x = x_test_mn
        test_y = y_test_mn

    class_count = len(set(train_y))
    print("Here is the set of all labels: ", set(train_y))

    if embed:
        (subset, subset_labels) = generate_null_data(type, dataset)

        x_train_new = np.concatenate((train_x, subset))
        y_train_new = np.concatenate((train_y, subset_labels))

        model = train_model(x_train_new, class_count, dataset)

        model.summary()
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        r = model.fit(x_train_new, y_train_new, validation_data=(test_x, test_y), epochs=epoch)

        (test_subset, test_subset_labels) = generate_null_data(type, dataset)
        loss, acc = model.evaluate(test_subset, test_subset_labels, verbose=0)
        #null_embed.verify_watermark(32, 32, 10, 6, chi_null=acc)
        print("Here is loss: ", loss)
        print("Here is accuracy; ", acc)
    else:
        model = train_model(train_x, class_count, dataset)

        model.summary()
        model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        r = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epoch)

        (subset, subset_labels) = generate_null_data(type, dataset)
        loss, acc = model.evaluate(subset, subset_labels, verbose=0)
        #null_embed.verify_watermark(32, 32, 10, 6, chi_null=acc)
        print("Here is loss: ", loss)
        print("Here is accuracy; ", acc)

def train_model(x_train: ndarray, class_count: int, dataset: str):
    inp = Input(shape=x_train[0].shape)
    out = inp
    fil_count_conv = 32
    fil_count_dense = 512
    convolutional_count = CONV_COUNT_CIF
    dense_count = DENSE_COUNT_CIF

    if dataset == "mnist":
        convolutional_count = CONV_COUNT_MN
        dense_count = DENSE_COUNT_MN

    # Trains CONV_COUNT convolutional layers. Doubles filter count and applies pooling every other layer.
    for i in range(convolutional_count):
        if i % 2 == 0 and dataset == "cifar-10":
            out = train_one_conv(out, fil_count_conv, dataset)
        else:
            out = train_one_conv(out, fil_count_conv, dataset)
            out = MaxPooling2D((2, 2))(out)
            fil_count_conv *= 2


    out = Flatten()(out)

    # Trains DENSE_COUNT dense layers. Applies dropout every layer.
    for i in range(dense_count):
        if i == dense_count - 1:
            out = train_one_dense(out, class_count, 'softmax')
        else:
            out = train_one_dense(out, fil_count_dense, 'relu')

    model = Model(inp, out)

    return model

def train_one_conv(inp, filter_count, dataset: str):
    if dataset == "cifar-10":
        out = Conv2D(filter_count, (3, 3), activation='relu', padding='same')(inp)
        return BatchNormalization()(out)
    else:
        out = Conv2D(filter_count, (5, 5), activation='relu', padding='same')(inp)
        return BatchNormalization()(out)

def train_one_dense(inp, filter_count, act):
    out = Dropout(0.2)(inp)
    return Dense(filter_count, activation=act)(out)

def get_dimensions(dataset: str):
    if dataset == "cifar-10":
        return (x_train_cif.shape[1], x_train_cif.shape[2])
    elif dataset == "mnist":
        return (x_train_mn.shape[1], x_train_mn.shape[2])

def generate_null_data(type: str, dataset: str):
    global fil
    global true_label
    train_x = x_train_cif
    train_y = y_train_cif

    if dataset == "mnist":
        train_x = x_train_mn
        train_y = y_train_mn
    
    if not fil:
        if dataset == "cifar-10":
            (fil, true_label) = null_embed.generate_ownership_watermark(32, 32, 10, 6)
        else:
            (fil, true_label) = null_embed.generate_ownership_watermark(28, 28, 10, 6)
    train_count = train_x.shape[0]

    # Creation of null embedded data:
    null_subset_size = int(train_count * 0.1)
    null_rand_indices = np.random.choice(train_count, null_subset_size, replace=False)
    null_subset = train_x[null_rand_indices]
    null_subset_labels = train_y[null_rand_indices]

    # Creation of true embedded data:
    true_subset_size = int(train_count * 0.03)
    remaining_indices = np.setdiff1d(np.arange(train_count), null_rand_indices)
    true_rand_indices = np.random.choice(remaining_indices, true_subset_size, replace=False)
    true_subset = train_x[true_rand_indices]
    true_subset_labels = np.full(true_subset_size, true_label)
    print("True subset labels first 10: ", true_subset_labels[:10])

    (x, y) = fil.get_pos()
    print(x, y)
    bits = fil.get_bit()

    if type == "square":
        null_subset = embed_square_filter(null_subset, bits, x, y, dataset)
        # true_subset = embed_square_filter(true_subset, bits, x, y, True)
    elif type == "random":
        null_subset = embed_random_filter(null_subset, bits, dataset)
        # true_subset = embed_random_filter(true_subset, bits, True)
    elif type == "peripheral":
        null_subset = embed_peripheral_filter(null_subset, bits, dataset)
        # true_subset = embed_peripheral_filter(true_subset, bits, True)
    elif type == "circular":
        null_subset = embed_circular_filter(null_subset, bits, x, y, dataset)
        # true_subset = embed_circular_filter(true_subset, bits, x, y, True)
    elif type == "triangular":
        null_subset = embed_triangular_filter(null_subset, bits, x, y, dataset)
        # true_subset = embed_triangular_filter(true_subset, bits, x, y, True)
    
    # return np.concatenate([null_subset, true_subset]), np.concatenate([null_subset_labels, true_subset_labels]), true_subset, true_subset_labels
    return null_subset, null_subset_labels

def embed_square_filter(subset: ndarray, bits: int, x: int, y: int, dataset: str, inv: bool = False):
    for image in subset:
        current_bits = bits
        for i in range(SQUARE_SIDE):
            for j in range(SQUARE_SIDE):
                current_bits, image = set_pixel_vals(current_bits, image, x + i, y + j, dataset, inv)

    return subset

def embed_random_filter(subset: ndarray, bits: int, dataset: str, inv: bool = False):
    pixel_count = SQUARE_SIDE ** 2
    (h, w) = get_dimensions(dataset)
    all_coordinates = [(x, y) for x in range(w) for y in range(h)]
    rand_coords = random.sample(all_coordinates, pixel_count)

    for image in subset:
        current_bits = bits
        for (x, y) in rand_coords:
            current_bits, image = set_pixel_vals(current_bits, image, x, y, dataset, inv)

    return subset

def embed_peripheral_filter(subset: ndarray, bits: int, dataset: str, inv: bool = False):
    pixel_count = SQUARE_SIDE ** 2
    (_, w) = get_dimensions(dataset)

    for image in subset:
        current_bits = bits
        x, y = 0, 0
        for _ in range(pixel_count):
            current_bits, image = set_pixel_vals(current_bits, image, x, y, dataset, inv)
            x += 1
            if x >= w:
                x = 0
                y += 1  
    
    return subset

def embed_circular_filter(subset: ndarray, bits: int, x: int, y: int, dataset: str, inv: bool = False):
    pixel_count = SQUARE_SIDE ** 2
    radius = np.sqrt(pixel_count / np.pi)
    floored_r = int(np.floor(radius))
    center_x, center_y = x + floored_r, y + floored_r

    for image in subset:
        current_bits = bits
        for i in range(x, (x + floored_r * 2)):
            for j in range(y, y + floored_r * 2):
                if np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) <= radius:
                    current_bits, image = set_pixel_vals(current_bits, image, i, j, dataset, inv)

    return subset

def embed_triangular_filter(subset: ndarray, bits: int, x: int, y:int, dataset: str, inv: bool = False):
    pixel_count = SQUARE_SIDE ** 2
    row_count = row_count_triangle(pixel_count)
    (h, w) = get_dimensions(dataset)

    # The checks below prevent triangle from exceeding boundaries of the image.
    if x + row_count >= w:
        x = w - row_count
    if y + row_count >= h:
        y = h - row_count
    
    for image in subset:
        current_bits = bits
        for i in range(row_count):
            for j in range(i):
                current_bits, image = set_pixel_vals(current_bits, image, x, y, dataset, inv)
    return subset

def row_count_triangle(n: int):
    return int(np.floor((-1 + np.sqrt(1 + 8 * n)) / 2))

def set_pixel_vals(bits: int, image: ndarray, x: int, y: int, dataset: str, inv: bool = False) -> tuple[int, ndarray]:
    current_bit = bits & 1
    bits = bits >> 1
    if dataset == "cifar-10":
        if current_bit ^ inv:
            image[x][y] = [LAMBDA, LAMBDA, LAMBDA]
        else:
            image[x][y] = [-LAMBDA, -LAMBDA, -LAMBDA]
    else:
        if current_bit ^ inv:
            image[x][y] = LAMBDA
        else:
            image[x][y] = -LAMBDA
    return bits, image

run_training(True, "random", 20, "mnist")
# generate_null_data("circular")


# Path set necessary in each instance of WSL.
# export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=${CUDNN_PATH}/lib  
# Path to this script: "\mnt\c\Users\kaana\GitRepos\rp_kaltinay\cifar_dnn.py"
# "/mnt/c/Users/kaana/GitRepos/rp_kaltinay/cifar_dnn.py"