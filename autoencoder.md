<h1 align="center">
autoencoder.py
</h1>
<h2 align="center">
Extracting features from the signals using neural networks
</h2>

### This method is often used in the monitoring of machining processes, as it can automatically detect the anomaly happening during the process.

### Such a ability is based on ***Autoencoder's*** *(AE)* innate **reconstruction capability**.

### But in this page, I will just showcase how to exploit AE to extract signals' characteristics.

### [What is AE?](https://www.jeremyjordan.me/autoencoders/ "link" )

Before we go further, let's load our datasets made by [waveMaker.py](waveMaker.md "link" ).

To focus on the topic, we only deal with the **3rd signals** recorded from all process runs.

If you want to see what this signal looks like, go check [detail of signal_processing.py](signal_processing.md "link" )

```
signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset', first_signal_minus=False)
sample_rate = int(20000/10)
y = np.genfromtxt('demo_y.csv', delimiter=',')
siganl_idx_demo = 3
signal_runs = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo)
time_runs = sigpro.pick_one_signal(signals_runs, signal_idx=0)
```

Let me talk about my AE model first. The **AE model** consists of two parts, which are ***encoder*** and ***decoder***.

For **encoder**, there are **3 dense layers**. 

As for **decoder**, there are **1 dense layer** and **1 output layer**.

The **lengths** of all **dense layers** can be adjusted with argument ***latent_dim*** within the constructor.

[Reference of constricting AE](https://www.tensorflow.org/tutorials/generative/autoencoder "link" )

```
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
```

The point of this page is **training an AE so that it** can **reconstruct the input signal well**.

So, I resize 3rd signals from all process to a fixed length first, as **AE require inputs in a unfied dimension**.

Then use **all resized signals** as **input** and **output** for AE.

The **loss** function and additional **performance metric** for the AE can be assigned as **arguments** in ***autoencoder.train_AE()***.

The 4 plots below show the **original signal**, **resized signal**, **signal encoded** by the trained AE, **signal decoded** by the trained AE, respectively.

```
import autoencoder as ae
signals_resize, time_resize = sigpro.signal_resize(signal_runs, time_runs, final_length=min([run.shape[0] for run in signal_runs]))
ae_model, ae_train_history = ae.train_AE(signals_resize, loss='mean_squared_error', metric='mean_absolute_error', shrink_rate=100)
encoded_signal = ae_model.encoder(signals_resize).numpy()
decoded_signal = ae_model.decoder(encoded_signal).numpy()

sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo], color_='royalblue', title='OG Signal')
sigplot.draw_signal(signals_resize[run_idx_demo], time_resize[run_idx_demo], color_='seagreen', title='Resized Signal')
sigplot.draw_signal(encoded_signal[run_idx_demo], color_='peru', title='Encoded Signal')
sigplot.draw_signal(decoded_signal[run_idx_demo], time_resize[run_idx_demo], color_='crimson', title='Decoded Signal')
```

![OG](image/AE_1.png)
![Resized](image/AE_2.png)
![Encoded](image/AE_3.png)
![Decoded](image/AE_4.png)

The AE did a pretty good job to reconstruct the overall trends of the signal. 

If the **reconstruction** result is **good enough**, then we say the **encoder** in the AE is **good** as well.

In ***[featureExtraction.md](featureExtraction.md "link" )***, I mentioned that we want to **extract signal features** so that we can use them to **predict the resultant quality** of the machining process.

A **good encoder** can help with the **extraction of signal features**, as it know how to **keep the most important information** within the signal so that decoder can reconstruct it properly.

Thus, you can try using the **AE-encoded signals** as signal features to predict the resultant quality of your machining process.
