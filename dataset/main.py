import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

DATADIR = "D:/Datasets/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)

        #Coverting categories to numerical value based on the index
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                #Read img and convert to array
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)#remove the last param for color

                #Resize the img array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))#no need to change this for color

                #Append the img array and respective class to the training_data array
                training_data.append([new_array, class_num])
            except Exception as e:
                pass



def main():
    create_training_data()
    print('Length of the training data is')
    print(len(training_data))

    #Shuffle the data -> this step is very import for accurate results
    random.shuffle(training_data)

    #Check if shuffled
    print('The shuffled training data is as follows')
    for sample in training_data[:10]:#print only the first 10 classes
        print(sample[1])

    X = []
    y = []

    #Two arrays one for features(img) and the other for respective array
    for features, label in training_data:
        X.append(features)
        y.append(label)

    #Convert X and y to numpy array
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)#for color last param will be 3
    y = np.array(y)

    #Save the data
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    #Test X
    print('Testing final output')
    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)
    print(X[1])
    pickle_in.close()

if __name__ == "__main__":
    main()