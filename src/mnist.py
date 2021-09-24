# 🌐 MINST Dataset
# https://deepai.org/dataset/mnist

import numpy as np
from matplotlib import pyplot as plt

from model import NN

# --- Our model

# First layer:      784 neurons (pixels of our image with gray-scale values from 0..255)
# Hidden Layer 1:   16 neurons
# Hidden Layer 2:   16 neurons
# Output Layer:     10 layers (representing number 0..9)

# Layer1 ------------------- Layer2 ------------------- Layer3 ------------------- Layer4
# 784 (28*28)                  16                        16                   10 (digits 0-9)              <- #neurons
# input layer               hidden layer 1          hidden layer 2              output layer

#           784*16 weights                16*16 weights           16*10 weights                            <- #weights
#               16 biases                    16 biases               10 biases                             <- #biases
#          weights_2, biases_2         weights_3, biases_3      weights_4, biases_4


########################### 🔍 Hyperparameters 🔍 ##############################
layer_sizes = [784, 16, 16, 10]
batch_size = 100
batches_count = 100  # no. of total batches to process
step_size = 0.0001
plot_cost_after_every_n_batches = 10


################################ MNIST NN ######################################
with open("C:/Users/domin/Desktop/DeepLearning/GradientDescent/mnist/train-images.idx3-ubyte", "r") as images,\
        open("C:/Users/domin/Desktop/DeepLearning/GradientDescent/mnist/train-labels.idx1-ubyte", "r") as labels:

    # --- Header (images)
    print('▶ Processing header')
    # big-endian integer (32-bit, 4 bytes)
    header_images = np.fromfile(images, dtype='>i', count=4)

    magic_number = header_images[0]
    img_count = header_images[1]
    row_count = header_images[2]
    col_count = header_images[3]
    pixel_count = row_count*col_count

    print(f"#images\t\t {img_count}")
    print(f"#rows\t\t {row_count}")
    print(f"#cols\t\t {col_count}")
    print(f"#pixels\t\t {pixel_count} ({row_count}*{col_count})")

    # --- Header (labels)
    # big-endian integer (32-bit, 4 bytes)
    header_labels = np.fromfile(labels, dtype='>i', count=2)
    magic_number = header_labels[0]
    label_count = header_labels[1]

    print(f"#labels\t\t {label_count}")

    # --- Prepare for training & start
    print()
    print('▶ Processing images')
    model = NN(layer_sizes, batch_size, step_size)

    # Set up plot
    costs = np.zeros(batches_count)
    plt.ion()
    x = np.linspace(0, batches_count, batches_count, endpoint=False)
    fig, ax = plt.subplots()
    ax.set_xlabel('Batch')
    ax.set_ylabel('Cost')
    ax.set_title('NN cost')
    graph = ax.plot(x, costs)[0]
    plt.draw()

    # Start training
    for batch in range(batches_count):
        # --- Process one batch
        print(f'⏺ Batch {batch+1}')

        cost = None
        for j in range(batch_size):
            # image_count = i * batch_size + j

            # Image
            img = np.fromfile(images, dtype=np.ubyte, count=28*28)
            # plt.imshow(img.reshape((28, 28)), cmap="gray")
            # plt.show()

            # Label
            label = np.fromfile(labels, dtype=np.ubyte, count=1)
            label = label[0]
            # print(f"Processing image with label {label}")

            # 🧠 Train model 🧠
            cost = model.train(img, label)
            # only not None for the last sample in the batch
        costs[batch] = cost

        # Plot cost
        if batch % (plot_cost_after_every_n_batches-1) == 0:
            graph.set_ydata(costs)
            plt.draw()
            ax.set_ylim([0, np.max(costs)])
            plt.pause(0.01)

    plt.ioff()
    plt.show()
