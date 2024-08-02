<br />
<div align="center">
  <h1 align="center">Data mining with time series</h1>
  <h2 align="center">The current repository is still in progress and the contents may change!</h2>
  <h3 align="center">
    A combination of signal analysis and machine-learning algorithms.
    <br />
  </p>
</div>
(English is not my first language, so it may be hard to read this page. Sorry (/ _ \).)

I originally made this repo. to deal with a cooperative education program I participated in during my study. This Python library contains .py files to perform **signal processing**, **constructing machine learning algorithms**, and **plotting signal spectra**. These files helped me a lot, and hopefully, they may be found useful in your projects as well.

In this repo., one data and a folder are examples of how to use the .py files contained. For a better understanding, I am gonna make an imaginary scenario in which these example files originate.

## Imaging a machining process that can be monitored by working conditions such as vibrations, temperature, or any other signals. The goal is to predict the resultant surface quality of the process w/ the captured signals.
First, the **empirical data** (I made it up) are created, which are **'demo_y.csv'** and **folder 'demonstration_signal_dataset'**. The former is the **recorded quality values of several machining processes**, and the latter contains their corresponding **signals (.csv) captured during processes**.

### **'WaveMaker.py'** describes the details of determining created signals and quality values. Please execute it first before moving on to other .py files.

## WaveMaker.py:

The created data have multiple attribute to be manipulated. Starting with sampling rate, recording duration, and the total amount of machining process conducted. The number can be changed at will.
```
sr = int(20000/10) # the sampling rate
time_total = 5 # the recording duration
num_run = 50 # the total amount of machining process conducted.
```

After that, the empirical data (containing sigals & quality values) is created in a loop.

In the loop, I use 'random_seed' variable to create numerical fluctuations on signals' attributes (such as frequency, time, amplitude, etc.) and the 'y' values, which resembles a noised recordings in real applicaitons. 

In my experience, the real-world datasets from manufacturers tends to be messy and DIRTY AS HECK, so such a processing is necessary to simulate the real situation.
```
dataset_sig = []
dataset_y = []
for run_idx in range(num_run):
    random_seed = np.random.uniform(0.1, 0.3)
    t = np.arange(0, time_total*(1+np.random.uniform(0,0.1)), 1/sr)
    print(f'final time = {t[-1]:.2f} | time length = {t.shape[0]:.2f}')
    noise = np.random.normal(0,1,t.shape[0])
    amplitude_1 = np.array([10, 2, 1]) * (1 + random_seed)
    amplitude_2 = np.array([6, 1, 0.5]) * (1 + random_seed)
    amplitude_3 = np.array([3, 2, 1]) * (1 + random_seed)
    sig1 = sinMaker(A = amplitude_1[0], W = 20, THETA = 10) + sinMaker(A = amplitude_1[1], W = 230, THETA = 5) + sinMaker(A = amplitude_1[2], W = 500, THETA = 90) + noise
    sig2 = sinMaker(A = amplitude_2[0], W = 10, THETA = 0) + sinMaker(A = amplitude_2[1], W = 100, THETA = 30) + sinMaker(A = amplitude_2[2], W = 900, THETA = 90) + noise
    sig3 = sinMaker(A = amplitude_3[0], W = 120, THETA = 30) + expMaker(amplitude_3[1], 1, 0) + expMaker(amplitude_3[2], 2, 6) + expMaker(amplitude_3[2], 1.5, 15) + noise
    run_content = np.concatenate((t.reshape(-1, 1), sig1.reshape(-1, 1), sig2.reshape(-1, 1), sig3.reshape(-1, 1)), axis=1)
    dataset_sig.append(run_content.T)
    
    y1 = (amplitude_1[0] + amplitude_2[1]) * (1+amplitude_3[2]) * (1 + random_seed)
    y2 = ((amplitude_1[0] * amplitude_3[1] + amplitude_1[2]) - amplitude_2[0] * amplitude_2[2]) * (1 + random_seed)
    y3 = amplitude_1[0] * (1+amplitude_3[0]) * (1+amplitude_3[2]) * amplitude_3[1] * (1 + random_seed)
    dataset_y.append(np.array([y1, y2, y3]))
dataset_y = np.array(dataset_y)
```



With signals and quality records gathered from processes, we want to examine their relations, as it is helpful for us to understand the mechanisms underlying the machining process. The major functionalities of the repo. can be divided into three topics, which are related to different .py files below:
### signal_processing.py: Processing the captured signals using time-domain, frequency-domain, or time-frequency-domain analyses.
### featureExtraction.py & autoencoder.py: Extracting features from the signals.
### cross-validation.py, correlation_analysis.py, and any .py with 'classPSO' as prefix: Analyzing the relations between extracted features and qualities, or constructing machine-learning models to predict the qualities based on input signals.
