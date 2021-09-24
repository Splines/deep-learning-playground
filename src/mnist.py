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
batch_size = 50
batches_count = 60000//batch_size  # no. of total batches to process
step_size = 0.001
plot_cost_after_every_n_batches = 20
epochs = 10


################################ MNIST NN ######################################
with open("C:/Users/domin/Desktop/DeepLearning/GradientDescent/mnist/train-images.idx3-ubyte", "rb") as images,\
        open("C:/Users/domin/Desktop/DeepLearning/GradientDescent/mnist/train-labels.idx1-ubyte", "rb") as labels:

    # --- Header (images)
    print('▶ Processing image header')
    # big-endian integer(32-bit, 4 bytes)
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

    # --- Data (images)
    buf_images = images.read(row_count * col_count * img_count)
    images_data = np.frombuffer(buf_images, dtype=np.uint8).astype(np.float32)
    images_data = images_data.reshape(img_count, row_count * col_count)

    # --- Header (labels)
    print('▶ Processing label header')
    # big-endian integer (32-bit, 4 bytes)
    header_labels = np.fromfile(labels, dtype='>i', count=2)
    magic_number = header_labels[0]
    label_count = header_labels[1]
    print(f"#labels\t\t {label_count}")

    # --- Data (labels)
    buf_labels = labels.read(label_count)
    labels_data = np.frombuffer(buf_labels, dtype=np.uint8).astype(np.int64)

    # --- Prepare for training & start
    print()
    print('▶ Processing images')
    model = NN(layer_sizes, batch_size, step_size)
    costs = np.zeros(batches_count)
    accuracies = np.zeros(batches_count)

    # --- Set up plot
    plt.ion()
    x = np.linspace(0, batches_count, batches_count)
    fig, ax = plt.subplots()
    ax.set_xlabel('Batch')
    ax.set_ylabel('Cost & Accuracy')
    ax.set_title('NN cost')
    ax.set_ylim([0, 1.3])
    ax.set_xlim([0, batches_count])
    graph_cost = ax.plot(x, costs, label='cost')[0]
    graph_accuracy = ax.plot(x, accuracies, label='accuracy')[0]
    ax.legend()
    plt.draw()

    # Start training
    for epoch in range(epochs):
        print(f'⭕⭕⭕ Epoch {epoch+1}/{epoch}')
        for batch in range(batches_count):
            # --- Process one batch
            print(f'⏺ Batch {batch+1}')

            correct_predictions_count = 0
            for j in range(batch_size):
                img_index = batch * batch_size + j
                # Image
                # img = np.fromfile(images, dtype=np.ubyte, count=28*28)
                img = images_data[img_index]
                # plt.imshow(img.reshape((28, 28)), cmap="gray")
                # plt.show()

                # Label
                # label = np.fromfile(labels, dtype=np.ubyte, count=1)
                label = labels_data[img_index]
                # label = label[0]
                # print(f"Processing image with label {label}")

                # 🧠 Train model 🧠
                is_prediction_correct, cost = model.train(img, label)
                if is_prediction_correct:
                    correct_predictions_count += 1
                # note that cost is only not None for the last sample in the batch
            costs[batch] = cost
            accuracies[batch] = correct_predictions_count / batch_size

            # Plot cost
            # live matplotlib updates: https://stackoverflow.com/a/16446688
            if batch % (plot_cost_after_every_n_batches-1) == 0:
                graph_cost.set_ydata(costs)
                graph_accuracy.set_ydata(accuracies)
                plt.draw()
                # ax.set_ylim([0, np.max(costs)])
                plt.pause(0.01)

    plt.ioff()
    plt.show()
