# Set de datos: MNIST

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Import dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# UNCOMMENT THESE LINES TO PLOT RANDOM IMAGES FROM INPUT DATA

# n_rand = 16
# # Select n_rand random images from x_train
# images = np.random.randint(0, x_train.shape[0], n_rand)
# # Plot those n_rand random images
# for i in range(n_rand):
#     img = x_train[images[i], :, :]  # Image index, image x-axis, image y-axis
#     plt.subplot(4, 4, i + 1)  # working with 3x3 plots, i + 1 -> plot's index
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#     plt.title(y_train[images[i]])
#
# plt.suptitle(f'{n_rand} images from MNIST dataset')
# plt.show()

# Data pre-processing
# Images must be flattened to put them into neural network
# This is done by taking a 2D-grid and transforming it into a 1D-vector
# So x_train_flat -> [0, 1, 2, ..., 59999] 60.000 images
# and x_train_flat[0] -> [0, 1, 2, ..., 783] 28x28 pixels for each image
# finally x_train_flat : [[0,...,783], ..., [0,...,783]]
x_train_flat = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_flat = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Vectors must be normalized to fall between 0 and 1
# 0 dark 1 light
x_train_flat = x_train_flat / 255.0
x_test_flat = x_test_flat / 255.0

# Finally, y_train & y_test into one-hot representation
n_classes = 10
y_train_norm = keras.utils.to_categorical(y_train, n_classes)
y_test_norm = keras.utils.to_categorical(y_test, n_classes)

# Model creation
# Input layer: Dimension of 784 (flattened image size)
# Hidden layer: 15 ReLu activation neurons
# Output layer: Softmax activation (multiclass classification) and 10 categories

np.random.seed(1)  # Restart random seed, could be omitted
input_dim = x_train_flat.shape[1]
output_dim = y_train_norm.shape[1]

neurons = 15
model = keras.Sequential()
model.add(keras.layers.Dense(neurons, input_dim=input_dim, activation='relu', name='layer_1'))
model.add(keras.layers.Dense(output_dim, activation='softmax', name='layer_2'))
print('Summary')
print(model.summary())

# Model compilation and training
# SGD, learning rate = 0.2, error function: cross entropy
# metrics: accuracy
# Note: softmax -> categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=0.2),
              metrics=['accuracy'])

# AS EPOCHS ARE INCREMENTED, YOU GET BETTER PRECISION
n_epochs = 200
b_size = 1024
story = model.fit(x=x_train_flat,
                  y=y_train_norm,
                  epochs=n_epochs,
                  batch_size=b_size,
                  verbose=2)

plt.subplot(1, 2, 1)
plt.plot(story.history['loss'])
plt.title('Perdida vs iteraciones')
plt.ylabel('Perdida')
plt.xlabel('Iteracion')

plt.subplot(1, 2, 2)
plt.plot(story.history['accuracy'])
plt.title('Precision vs iteraciones')
plt.ylabel('Precision')
plt.xlabel('Iteracion')

plt.show()

# Calculate model precision with test set
score = model.evaluate(x_test_flat, y_test_norm, verbose=0)  # 0 means show nothing
print(f'Precision rate: {100 * score[1]:.1f}%')

# Make some predictions
y_pred = model.predict(x_test_flat)
# Random select n_rand images from flattened images
n_rand = 9
images = np.random.randint(0, x_test_flat.shape[0], n_rand)

for i in range(len(images)):
    idx = images[i]
    img = x_test_flat[idx, :].reshape(28, 28)
    original_cat = np.argmax(y_test_norm[idx, :])
    predicted_cat = np.argmax(y_pred[idx])
    # Plot the predicted results
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'{original_cat} classified as: {predicted_cat}')

plt.suptitle('Classifications in test set')
plt.show()
