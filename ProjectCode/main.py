#IMPORT LIBRARY
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#MAKE CLASS CALLBACK
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            # Stop if threshold is met
            print("\naccuracy higher than 0.99  so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

def siaga_model():
    #READ DATASET
    fire = pd.read_csv("./fire_dataset_random.csv")

    selected_parameter = ['Temperature', 'MQ139', 'Detector']
    parameter = np.array(fire[selected_parameter].values)
    labels = np.array(fire['Fire Alarm'])

    # SIZE = len(fire)
    TRAINING = 10000
    VALID = 900 + TRAINING

    train_param, valid_param, test_param = parameter[:TRAINING], parameter[TRAINING:VALID], parameter[VALID:]
    train_label, valid_label, test_label = labels[:TRAINING], labels[TRAINING:VALID], labels[VALID:]

    #MAKE MODEL
    model = tf.keras.models.Sequential([
        keras.layers.Dense(units=512, input_shape=(3,), activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=3, activation='softmax')
    ])

    #COMPILE MODEL
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    #FIT MODEL
    history = model.fit(train_param,
              train_label,
              validation_data=(valid_param, valid_label),
              epochs=100)

    #TEST MODEL
    benar,salah = 0, 0
    print("!===============================!")
    hasil = np.argmax(model.predict([test_param]),axis=1)
    hasil_df ={
        "Temperature":test_param[:, 0],
        "MQ139": test_param[:, 1],
        "Detector": test_param[:, 2],
        "Fire Alarm": test_label,
        "Hasil": hasil
    }
    hasil_df = pd.DataFrame(hasil_df)
    # pd.set_option('display.max_rows', None)
    print(hasil_df)

    for i in range(0,len(test_param)):
        if hasil[i] == int(test_label[i]):
            benar +=1
        else:
            salah += 1
    print(f"Benar = {benar}; Salah = {salah}; Accuracy: {benar/float(benar+salah)}")


    #RETRIEVE A LIST OF RESULTS
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    #PLOT ACCURACY
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.show()
    print("")

    #PLOT LOSS
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.show()

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = siaga_model()
    # model.save("siaga_model.h5")