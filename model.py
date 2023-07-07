from tradeBot import TradeBot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

    if single_step:
        labels.append(target[i+target_size])
    else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

class Model:
    def __init__(self, data, future=1, past_history=30, ):
        self.TRAIN_SPLIT = 300
        self.BATCH_SIZE = 20 # размер одного батча для тренировки
        self.BUFFER_SIZE = 400 # размер буфера для перемешивания данных
        self.EVALUATION_INTERVAL = 200 # количество тренировок в 1 эпоху
        self.EPOCHS = 20

        self.past_history = past_history
        self.future_target = future
        self.STEP = 1
        self.model = tf.keras.models.load_model('/home/python/trade_bot/best_model')

        # параметры нормализации
        self.data_mean = data[:self.TRAIN_SPLIT].mean(axis=0)
        self.data_std = data[:self.TRAIN_SPLIT].std(axis=0)   

    def train(self, dataset: np.array) -> None:
        dataset = self.__normalize(dataset)
        train_data, val_data = self.__train_and_val(dataset)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8, activation=tf.keras.activations.relu),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.LSTM(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer="adam", loss=tf.keras.losses.MSLE)


        self.model.fit(train_data, epochs=self.EPOCHS,
                        steps_per_epoch=self.EVALUATION_INTERVAL,
                        validation_data=val_data,
                        validation_steps=5)
        


    def __normalize(self, dataset: np.array):
        data_mean = dataset[:self.TRAIN_SPLIT].mean(axis=0)
        data_std = dataset[:self.TRAIN_SPLIT].std(axis=0)

        dataset = (dataset-data_mean)/data_std

        return dataset
    
    def __train_and_val(self, dataset: np.array) -> tf.data.Dataset:
        x_train, y_train = multivariate_data(dataset, dataset[:, 0], 0,
                                                        self.TRAIN_SPLIT, self.past_history,
                                                        self.future_target, self.STEP,
                                                        single_step=True)
        x_val, y_val = multivariate_data(dataset, dataset[:, 0],
                                                        self.TRAIN_SPLIT, None, self.past_history,
                                                        self.future_target, self.STEP,
                                                        single_step=True)

        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.batch(self.BATCH_SIZE).repeat()
        
        return train_data, val_data
    
    def predict(self, x: list) -> float:
        x = (x-self.data_mean)/self.data_std
        pred = tf.keras.utils.pad_sequences([x for _ in range(20)], dtype="float64")
        return self.model.predict(pred)[0]
    
    def save(self, path="./new_model"):
        self.model.save(path)
    
