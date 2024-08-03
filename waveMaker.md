<h1 align="center">
waveMaker.py
</h1>
The created data have multiple attribute to be manipulated. Starting with sampling rate, recording duration, and the total amount of machining process conducted. The number can be changed at will.


```
sr = int(20000/10) # the sampling rate
time_total = 5 # the recording duration
num_run = 50 # the total amount of machining process conducted.
```

After that, the empirical data (containing **sigals & quality values**) is created in a loop.

In the loop, I use ***'random_seed'*** variable to create **numerical fluctuations** on signals' attributes (such as frequency, time, amplitude, etc.) and the 'y' values, which **resembles a noised recordings in real applicaitons**. 

*In my experience, the real-world datasets from manufacturers tends to be MESSY and DIRTY AS HECK, so such a processing is necessary to simulate the real situation.*

### Any variable named after ***'amplidtude'*** control the **amplitudes of signals**.

### Any variable named after ***'sig'*** are **signals recorded** during the machining process (in this imaginary scenario).

The recorded signals are composed of serveral signal components. The signal components can be defined by the function ***sinMaker*** and ***expMaker***, and variable ***noise***
```
def sinMaker(A, W, THETA):
    # A: amplitude; W: Hz; THETA: phase angle
    return A * np.sin((W * 2*np.pi) * t + THETA)

def expMaker(A, G, Tau, isGaussian=True):
    # A: amplitude of exp()
    # G: growing value
    # Tau: time shift (>0: lag, <0: ahead)
    newTimeVector = G * (t - Tau)
    if isGaussian:
        newTimeVector = newTimeVector - 0.5 * (t + -1 * Tau) ** 2
    return A * np.exp(newTimeVector)

noise = np.random.normal(0,1,t.shape[0])
```

### Any variable named after ***'y'*** are **resultant quality** of the current machining process (in this imaginary scenario). 

In the code below, it can be seen that I respectively define **3** different **signals** and **y value** for each run in the loop.

**Signals of processing runs** consists a distinct sets of components with ***fluctuated amplitude*** and ***fixed frequency***. And they are assigned to variable ***dataset_sig***

**y values**, aka **surface qualities**, are defined by the ***amplitudes of signals*** and ***random_seed***. And they are assigned to variable ***dataset_y***
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
***dataset_sig*** should be the list containing ***num_run*** numpy arrays, and each array have **4 rows** **(time, sig1, sig2, sig3)** and columns with inconsistent amounts, as the **signal lengths are different**.

BTW, when each run's data is created, it prints the recorded length of that run: 
```
final time = 5.21 | time length = 10411.00
final time = 5.30 | time length = 10600.00
final time = 5.45 | time length = 10900.00
final time = 5.06 | time length = 10121.00
...
...
...
```

Now that we have the whole datasets, which are dataset_sig and dataset_y, we can save them as .csv.
```
def save_files(folder, data_sig, data_y):
    for run_idx, run in enumerate(data_sig):
        if run_idx < 10:
            np.savetxt(f'.\\{folder}\\demo_signals_0{run_idx}.csv', run.T, delimiter=',')
        else:
            np.savetxt(f'.\\{folder}\\demo_signals_{run_idx}.csv', run.T, delimiter=',')
    np.savetxt(f'.\\demo_y.csv', data_y, delimiter=',')

save_files('demonstration_signal_dataset', dataset_sig, dataset_y)
```

The csv files should be located like this:
```
${Data-Mining-w-Time-Series-For_demonstration}
├── demo_y.csv
├── demonstration_signal_dataset
    ├── demo_signals_00.csv
    ├── demo_signals_01.csv
    ├── demo_signals_02.csv
    ...
    ...
```
