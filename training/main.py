import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import matplotlib.pyplot as plt

NAME = "cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))


def visualize(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def trainModel(X, y):
    # Callbacks
    tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME))

    # Normalize data (0 - 1)
    X = X/255.0

    # Build the model
    model = Sequential()

    # 1st layer
    # Convulation 64 is the size and the filter is 3x3
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    # Activation
    model.add(Activation("relu"))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd layer
    # Convulation
    model.add(Conv2D(64, (3, 3)))
    # Activation
    model.add(Activation("relu"))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Final layer
    model.add(Flatten())  # Convert 3D features map to 1D feature vectors
    #model.add(Dense(64))  # Fully Connected layer
    #model.add(Activation('relu'))

    model.add(Dense(1))  # Fully connected layer
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=['accuracy'])

    # Train the model, here 10% of data shall be user for validation
    epochs = 10
    history = model.fit(X, y, batch_size=32, epochs=epochs,
                        validation_split=0.1, callbacks=[tensorboard])
    print(history)
    visualize(history, epochs)


def main():
    # Load previously prepared data
    X = pickle.load(open("../dataset/X.pickle", "rb"))
    y = pickle.load(open("../dataset/y.pickle", "rb"))

    # Train the model
    trainModel(X, y)


if __name__ == "__main__":
    main()
