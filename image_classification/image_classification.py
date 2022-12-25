# Image classification based on
# https://keras.io/examples/vision/image_classification_from_scratch/
# Dataset: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

# JPEG image files on disk, no leveraging pre-trained weights or
# pre-made Keras Application model
# Use of Keras layers for image standarization and image augmentation

import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

# Define the model
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.1)
])


def filter_corrupted_data():
    removed_images = 0

    for folder_name in ('Cat', 'Dog'):
        folder_path = os.path.join('datasets/PetImages', folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)

            with open(fpath, 'rb') as fobj:
                is_jfif = tf.compat.as_bytes('JFIF') in fobj.peek(10)

            if not is_jfif:
                removed_images += 1
                # Remove corrupted image
                os.remove(fpath)
    print(f'Removed from dataset: {removed_images}')


# Generate a dataset
def generate_data(image_size, batch_size):
    train_ds, val_ds = keras.preprocessing.image_dataset_from_directory(
        'datasets/PetImages',
        validation_split=0.2,  # fraction of data to reserve for validation
        subset='both',  # returns a tuple of two datasets (the training and validation datasets respectively)
        seed=1337,  # Optional random seed for shuffling and transformations
        image_size=image_size,
        batch_size=batch_size
    )

    return train_ds, val_ds


# Visualize data
def visualize_data(train_images, is_augmented_mode=False):
    plt.figure(figsize=(10, 10))
    for images, labels in train_images.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            if not is_augmented_mode:
                plt.imshow(images[i].numpy().astype('uint8'))
            else:
                augmented_images = data_augmentation(images)
                plt.imshow(augmented_images[i].numpy().astype('uint8'))
            plt.title(int(labels[i]))
            plt.axis('off')

    plt.suptitle('Data from dataset')
    plt.show()


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Defining output
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # Project residual
        residual = keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train_model(model: keras.Model, train_ds, val_ds):
    epochs = 25
    callbacks = [
        keras.callbacks.ModelCheckpoint(f'save_at_{epochs}.keras')
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds
    )


def main():
    image_size = (180, 180)
    batch_size = 128

    filter_corrupted_data()
    train_ds, val_ds = generate_data(image_size, batch_size)
    visualize_data(train_ds)
    visualize_data(train_ds, True)

    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    # keras.utils.plot_model(model, show_shapes=True)

    train_model(model, train_ds, val_ds)

    # Try new data
    img = keras.preprocessing.image.load_img(
        "datasets/PetImages/Cat/6779.jpg", target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


if __name__ == '__main__':
    main()
