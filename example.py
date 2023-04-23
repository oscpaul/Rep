import keras
from keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import time
from keras.metrics import Recall
from keras.metrics import Precision

print("\nPlease enter a name for the exported model.")
print("The .h5 extension will be added for you.")
modelName = input()
modelName += ".h5"
startTime = time.time()

# Important: The model gets torn down after each k-run, so only 
# the final run will be exported. If you are not testing and are 
# trying to produce one model to work with, set k to 1. This loop 
# exists because it may be useful to run tests by creating and 
# evaluating several models given the same parameters. 

# Number of runs for model creation and training.
# If k=5, 5 models will be created and tested.
# Only the last one will be exported. If k=1, only 1 model will 
# be created, tested, and exported.
k=1
i=0
accuracyList = []
recallList = []
precisionList = []
f1List = []

while(i<k):
    #generate dataset
    path_root = "train400" # dataset to train from

    # target_size is the size of the image piece to use.
    # Do not change this as it will make your model incompatible
    # with the grading set.
    # batch_size is the number of samples to use for each round of training.
    # Feel free to try different values here to see how that affects the models.
    # train400 set contains nearly 400 samples for every malware family. It
    # has 3198 samples total.
    batches = ImageDataGenerator().flow_from_directory(directory=path_root, target_size=(64,64), batch_size=10)
    print(batches.class_indices)

    imgs, labels=next(batches)

    # Split into train and test. You may try different ratios for train_size and test_size.
    # Here, 0.9 for train_size gives 90% of the samples to train on, 0.1 for test_size will
    # use the other 10% for testing. These values should add up to 1.0, more or less than that
    # could result in undefined behaviors.
    X_train, X_test, y_train, y_test = train_test_split(imgs/255, labels, train_size=0.9, test_size=0.1)
    print("X_train.shape")
    print(X_train.shape)
    print("X_test.shape")
    print(X_test.shape)
    print("y_train.shape")
    print(y_train.shape)
    print("y_test.shape")
    print(y_test.shape)

    # Denotes number of malware families, don't change.
    num_classes=8

    # Here we have defined a CNN model.
    # One way to increase the accuracy of a model is to add 
    # more layers. I would suggest trying this below the 
    # "Dropout(0.5) line. Layers cannot accept larger outputs, 
    # so below Dense layer 128 there cannot be another dense 
    # layer with a value larger than 128. You can make the 
    # consecutive layers smaller, however. A dense layer of 128 
    # can be followed by a smaller layer (like 64, then 32 etc). 
    # These do not have to follow a pattern but it may produce 
    # good results to try a sequence of layers that does.
    def malware_model():
        Malware_model = Sequential()
        Malware_model.add(Conv2D(30, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(64,64,3)))
        Malware_model.add(MaxPooling2D(pool_size=(2,2)))
        Malware_model.add(Conv2D(15, (3, 3), activation='relu'))
        Malware_model.add(MaxPooling2D(pool_size=(2,2)))
        Malware_model.add(Dropout(0.25))
        Malware_model.add(Flatten())
        Malware_model.add(Dense(128, activation='relu'))
        Malware_model.add(Dropout(0.5))
        #Malware_model.add(Dense(64, activation='relu'))
        Malware_model.add(Dense(50, activation='relu'))
        #Malware_model.add(Dense(32, activation='relu'))
        #Malware_model.add(Dense(16, activation='relu'))
        #Malware_model.add(Dense(8, activation='relu'))
        #Malware_model.add(Dropout(0.25))
        Malware_model.add(Dense(num_classes, activation='softmax'))
        Malware_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return Malware_model

    Malware_model = malware_model()
    # The .summary() function here is useful for
    # displaying information about the created model.
    Malware_model.summary()

    # Train. The number of epochs here are the number of 
    # rounds to train for. Greater numbers of epochs could 
    # lead to better training, but too many can lead to 
    # overfitting. Try different values for the number of epochs.
    Malware_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    # test
    scores = Malware_model.evaluate(X_test, y_test)
    print('Final CNN accuracy for round ', i+1, ":", scores[1])
    accuracyList.append(scores[1])
    i += 1

#end while loop for k creations of models
print("\nEnd of k average loop\n\nAccuracyList:")
endTime = time.time()
min = (endTime - startTime) / 60 # time in minutes
average=0
x=0
while(x<k):
    print("Round", x+1, ":", accuracyList[x])
    average += accuracyList[x]
    x+=1
average = average / k
print("\nFinal average for", k, "runs is", average)
print("Execution time:", min, "minutes")

Malware_model.save(modelName)
print("Model saved to", modelName)
