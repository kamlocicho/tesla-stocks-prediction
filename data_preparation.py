import numpy as np

class DataPreparation:
    def __init__(self, data_scaled, window=70):
        self.data_scaled = data_scaled
        self.window = window

    def prepare_data(self):
        x_train = []
        y_train = []

        for i in range(self.window, 2900):
            x_train.append(self.data_scaled[i-self.window:i, 0])
            y_train.append(self.data_scaled[i,0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_test = []
        y_test = []
        for i in range(2901, 3382):
            x_test.append(self.data_scaled[i-self.window:i, 0])
            y_test.append(self.data_scaled[i,0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        return x_train, y_train, x_test, y_test