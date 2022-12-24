import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

definitions = {
    0: "t-shirt/top",
    1: "trousers/pants",
    2: "pullover shirt",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boots"
}

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

n_rand = 16
images = np.random.randint(0, x_train.shape[0], n_rand)

for i in range(n_rand):
    img = x_train[images[i], :, :]
    plt.subplot(4, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(definitions[y_train[images[i]]])

plt.suptitle(f'{n_rand} images from Fashion MNIST dataset')
plt.show()

x_train_flat = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_flat = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

x_train_flat = x_train_flat / 255.0
x_test_flat = x_test_flat / 255.0

n_classes = 10
y_train_norm = keras.utils.to_categorical(y_train, n_classes)
y_test_norm = keras.utils.to_categorical(y_test, n_classes)

# Model
np.random.seed(1)
input_dim = x_train_flat.shape[1]
output_dim = y_train_norm.shape[1]

neurons = 15
model = keras.Sequential()
model.add(keras.layers.Dense(neurons, input_dim=input_dim, activation='relu', name='input_layer'))
model.add(keras.layers.Dense(output_dim, activation='softmax', name='output_layer'))

print('Summary')
print(model.summary())

# Model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=0.2),
              metrics=['accuracy'])

n_epochs = 300
b_size = 1000
story = model.fit(x_train_flat,
                  y_train_norm,
                  batch_size=b_size,
                  epochs=n_epochs,
                  verbose=3)

plt.subplot(1, 2, 1)
plt.plot(story.history['loss'])
plt.title('Loss v. Iterations')
plt.ylabel('Loss')
plt.xlabel('Iteration')

plt.subplot(1, 2, 2)
plt.plot(story.history['accuracy'])
plt.title('Accuracy v. Iterations')
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.show()

# Calculate model accuracy
score = model.evaluate(x_test_flat, y_test_norm, verbose=0)
print(f'Precision rate: {100 * score[1]:.1f}%')

# Make predictions
y_pred = model.predict(x_test_flat)

n_rand = 16
images = np.random.randint(0, x_test_flat.shape[0], 16)

for i in range(len(images)):
    idx = images[i]
    img = x_test_flat[idx, :].reshape(28, 28)
    original_cat = np.argmax(y_test_norm[idx, :])
    predicted_cat = np.argmax(y_pred[idx])

    plt.subplot(4, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'{definitions[original_cat]} classified as: {definitions[predicted_cat]}')

plt.suptitle('Classifications in test set')
plt.show()
