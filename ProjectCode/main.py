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
    fire = pd.read_csv("./fire_dataset.csv").sample(frac=1).reset_index(drop=True)

    selected_parameter = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]']

    parameter = np.array(fire[selected_parameter].values)
    labels = np.array(fire['Fire Alarm'])

    #print(parameter)
    #print(categories)

    #VARIABEL
    train_portion = 0.8
    valid_portion = 0.1
    test_portion = 0.1
    SIZE = len(fire)
    TRAINING = int(SIZE*train_portion)
    VALID = int(SIZE*valid_portion)+TRAINING

    train_param, valid_param, test_param = parameter[:TRAINING],parameter[TRAINING:VALID],parameter[VALID:]
    train_label, valid_label, test_label = labels[:TRAINING],labels[TRAINING:VALID],labels[VALID:]


    model = tf.keras.models.Sequential([
        keras.layers.Dense(units=256,input_shape=(4,),activation='relu'),
        keras.layers.Dense(units=64,activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    history = model.fit(train_param,
              train_label,
              epochs=69,
              validation_data=(valid_param, valid_label),
              callbacks=[callbacks])

    benar,salah = 0,0
    hasil = int(model.predict([test_param])[0][0])
    print("===============================")
    hasil_df ={
        "Temperature[C]":test_param[:,0],
        "Humidity[%]": test_param[:, 1],
        "TVOC[ppb]": test_param[:, 2],
        "eCO2[ppm]": test_param[:, 3],
        "kategori": test_label,
        "Hasil": hasil
    }
    hasil_df = pd.DataFrame(hasil_df)
    print(hasil_df)

    for i in range(0,len(test_param)):
        if hasil == int(test_label[i]):
            benar +=1
        else:
            salah += 1
    print(f"Benar = {benar}; Salah = {salah}; Accuracy: {benar/float(benar+salah)}")
    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.show()
    print("")

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
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