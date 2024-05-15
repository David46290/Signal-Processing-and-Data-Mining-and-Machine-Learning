import numpy as np
import tensorflow as tf
from keras import optimizers as opti
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping

 

class Autoencoder(Model):
      def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Dense(latent_dim, input_shape=shape, activation='relu'),
            # layers.Dropout(0.1),
            layers.Dense(latent_dim//2, activation='relu'),
            # layers.Dropout(0.1),
            # layers.Dense(latent_dim//10, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dim, activation='relu'),
            # layers.Dropout(0.1),
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='linear'),
            layers.Reshape(shape)
        ])
    
      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def build_AE(x_input, loss, metric, shrink_rate=10):
    optimizer = opti.Adam(learning_rate=0.005)
    shape = x_input.shape[1:]
    latent_dim = shape[0] // shrink_rate
    model = Autoencoder(latent_dim, shape)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model(tf.ones((1, shape[0]))) # initialize weights
    return model

def train_AE(x_input, loss='mean_absolute_error', metric='cosine_similarity', shrink_rate=10):
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    ae_model = build_AE(x_input, loss, metric, shrink_rate)
    history = ae_model.fit(x_input, x_input,
                epochs=1000, batch_size=10,
                shuffle=True, verbose=0, callbacks=[callback])
    return ae_model, history

def AGWN_data_expansion(train_signals, expansion_coeff):
    """
    Apply additive gaussian noise to signals
    Chose signals from random runs to add noise
    
    Input: 
        train_signals: ndarray (num_sample, signal_length)
        
    Output:
        new_signals: ndarray (num_sample * (1+expansion_coeff), signal_length)
    """
    num_sample = train_signals.shape[0]
    len_sample = train_signals.shape[1]
    num_sample_add = int(num_sample * expansion_coeff)
    train_signals_new = []
    for idx in range(0, num_sample_add):
        run_idx = np.random.randint(num_sample)
        noised_signal = train_signals[run_idx] +  np.random.normal(0,1,len_sample) # mean:0, std:1, length: len_sample
        train_signals_new.append(noised_signal)
    train_signals_new = np.array(train_signals_new)
    new_signals = np.concatenate((train_signals, train_signals_new), axis=0)
    return new_signals
        
