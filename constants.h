#pragma once

#define ROOT_DIR "D:/licenta/ReteleGenerative/"

#define MNIST ROOT_DIR "MNIST/"
#define MNIST_TRAIN_IMAGES MNIST "train-images.idx3-ubyte"
#define MNIST_TRAIN_LABELS MNIST "train-labels.idx1-ubyte"
#define MNIST_TEST_IMAGES MNIST "t10k-images.idx3-ubyte"
#define MNIST_TEST_LABELS MNIST "t10k-labels.idx1-ubyte"

#define PY_BRIDGE ROOT_DIR "PyBridge/"
#define PY_TRAIN PY_BRIDGE "data/train/"
#define PY_RESULTS PY_BRIDGE "data/results/"

#define PY_BRIDGE_LINEARLY_SEPARABLE_DATA PY_TRAIN "linearly_separable.txt"
#define PY_BRIDGE_LINEAR_MODEL PY_RESULTS "linear_separator_model.txt"

#define PY_BRIDGE_MNIST_LAYER PY_RESULTS "mnist_layer.txt"

#define PY_BRIDGE_XOR_DATA PY_TRAIN "xor.txt"
#define PY_BRIDGE_XOR_MODEL PY_RESULTS "xor_model.txt"
