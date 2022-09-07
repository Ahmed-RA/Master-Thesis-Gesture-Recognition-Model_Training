import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import sklearn as sk
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt





def build_model(hp):
        model = keras.Sequential()
        model.add(layers.LSTM(units=hp.Int(f"units_{1}", min_value=128, max_value=512, step=32),
                            input_shape = (14 , 33), 
                            return_sequences = True))
        counter = 0
        for i, v in enumerate(range(2, hp.Int("num_layers", 3, 4))):
            model.add(
                layers.LSTM(                    
                    units=hp.Int(f"units_{v}", min_value=128, max_value=512, step=32),
                    return_sequences=True
                )
            )
            counter = v + 1

        model.add(
                layers.LSTM(
                    units=hp.Int(f"units_{counter}", min_value=128, max_value=512, step=32),
                ))
        model.add(layers.Dense(6, activation="softmax"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        return model


   







def plot_acc_loss(history, model_name, conf_matrix, conf_matrix_test):
    f1 = plt.figure()
    plt.plot(history.history['categorical_accuracy'], 'r', label='Training accuracy')
    plt.plot(history.history['val_categorical_accuracy'], 'g', label='Validation accuracy')
    plt.grid(visible=True)
    plt.yticks(np.arange(0.5,1,0.05))
    plt.title('Training Vs Validation Accuracy')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.getcwd() + "\\" + "kt_tuned_models_results\\" + "Accuracy_" + model_name + ".png") #todo: dynamic paths from cmdline

    f2 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(visible=True)
    plt.yticks(np.arange(0,1,0.1))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(os.getcwd() + "\\" + "kt_tuned_models_results\\" + "Loss_" + model_name  + ".png")

    f3 = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0,1,2,3,4,5])
    f3.plot()

    diag_sum = 0
    for i in range(conf_matrix.shape[0]):
        diag_sum += conf_matrix[i][i]

    print("confusion matrix accuracy on validation data is " + str(diag_sum/conf_matrix.sum()))
    print("confusion matrix error on validation data is " + str(1 - (diag_sum/conf_matrix.sum())))

    plt.savefig(os.getcwd() + "\\" + "kt_tuned_models_results\\" + "ConfMat_val_" + model_name  + ".png")

    f4 = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=[0,1,2,3,4,5])
    f4.plot()

    diag_sum = 0
    for i in range(conf_matrix_test.shape[0]):
        diag_sum += conf_matrix_test[i][i]

    print("confusion matrix accuracy on test data is " + str(diag_sum/conf_matrix_test.sum()))
    print("confusion matrix error on test data is " + str(1 - (diag_sum/conf_matrix_test.sum())))

    plt.savefig(os.getcwd() + "\\" + "kt_tuned_models_results\\" + "ConfMat_test_" + model_name  + ".png")



if __name__=="__main__":

    

    train_data = np.load("Data\\train_fpfh_data.npy")
    train_labels = np.load("Data\\train_fpfh_labels.npy")

    test_data = np.load("Data\\test_fpfh_data.npy")
    test_labels = np.load("Data\\test_fpfh_labels.npy")

    val_data = np.load("Data\\val_fpfh_data.npy")
    val_labels = np.load("Data\\val_fpfh_labels.npy")

    X_train_tensor = tf.constant(train_data)
    X_test_tensor = tf.constant(test_data)
    X_val_tensor = tf.constant(val_data)

    y_train_tensor = tf.constant(train_labels)
    y_test_tensor = tf.constant(test_labels)
    y_val_tensor = tf.constant(val_labels)
    
    
    
    batch_sizes = [16, 32, 48, 64]
    best_hps = []

    tuner = keras_tuner.Hyperband(
            hypermodel=build_model,
            objective="val_categorical_accuracy",
            max_epochs=30,
            hyperband_iterations=1,
        )

    for i in batch_sizes:
        
        tuner.search(X_train_tensor, y_train_tensor, epochs=30, batch_size = i, validation_data=(X_val_tensor, y_val_tensor), 
                callbacks=[tf.keras.callbacks.EarlyStopping('val_categorical_accuracy', patience=3)])


        best_hps.append((tuner.get_best_hyperparameters()[0],i))


    print("[INFO] Retraining best 4 models...")
    
    counter = 1
    for hp, b_s in best_hps:
        model = build_model(hp)

        
        history =  model.fit(X_train_tensor, y_train_tensor, epochs = 30, 
                            batch_size = b_s, verbose = 1, validation_data = (X_val_tensor, y_val_tensor)) 

        #predicting test data and validation data to construct confusion matrix

        predictions = model.predict(X_val_tensor)
        predictions_reduced = tf.math.argmax(predictions, 1).numpy()  
        vallabel_reduced = tf.math.argmax(y_val_tensor, 1).numpy()
            
        conf_matrix = sk.metrics.confusion_matrix(vallabel_reduced, predictions_reduced, labels=[0,1,2,3,4,5])

        predictions_test = model.predict(X_test_tensor)
        predictions_test_reduced = tf.math.argmax(predictions_test, 1).numpy()  
        testlabel_reduced = tf.math.argmax(y_test_tensor, 1).numpy()
            
        conf_matrix_test = sk.metrics.confusion_matrix(testlabel_reduced, predictions_test_reduced, labels=[0,1,2,3,4,5])

        plot_acc_loss(history, "fpfh_model_batchsize"+ str(b_s) , conf_matrix, conf_matrix_test)

        np.save("confmat_val_fpfh_model_batchsize" + str(b_s) + ".npy", conf_matrix)
        np.save("confmat_test_fpfh_model_batchsize" + str(b_s) + ".npy", conf_matrix_test)

        model.save('fpfh_model_batchsize' + str(b_s) + '.h5')
        counter += 1

    print("end")