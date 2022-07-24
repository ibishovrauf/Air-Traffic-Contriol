import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error

class Model:
    def __init__(self, state_size,
                 action_size,
                 width,
                 layer_nums,
                 lr,
                 test
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.width = width
        self.layer_nums = layer_nums
        self.model = self._build_model()
        self.lr = lr
        self.test = test

    def _build_model(self):

        inputs = Input(shape=(self.state_size,))
        x = Dense(self.width, activation='relu')(inputs)
        for _ in range(self.layer_nums):
            x = Dense(self.layer_nums, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=inputs,outputs=outputs)
        optimizer = Adam(lr=self.lr)
        model.compile(loss=mean_squared_error, optimizer=optimizer)
        if self.test:
            model.load_weights('atc_weights')
        return model

    def train(self, x, y, n_epochs = 1, batch_size = 1):
        self.model.fit(x, y, batch_size=batch_size, epochs=n_epochs)

    def predict(self, state):
        return self.model.predict(state)

    def save_model(self):
        self.model.save('atc_model')





