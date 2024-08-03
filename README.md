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

<h2 align="center">Imaging a machining process that can be monitored by working conditions such as vibrations, temperature, or any other signals. The goal is to predict the resultant surface quality of the process with the captured signals.</h2>

First, the **empirical data** (I made it up) are created, which are **'demo_y.csv'** and **folder 'demonstration_signal_dataset'**. The former is the **recorded quality values of several machining processes**, and the latter contains their corresponding **signals (.csv) captured during processes**.

### **'waveMaker.py'** describes the details of determining created signals and quality values. Please execute it first before moving on to other .py files.

[Detail of waveMaker.py](waveMaker.md "link" )

**After executing waveMaker.py**, created datasets should be located like this:
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

Okay, so that is the end of the construction of the imaginary datasets. In **real situations**, you obtain the datasets **WITHOUT knowing how they are constructed**.

<h2 align="center">
So let's just forget how we build our datasets, and assuming we gathered those thing from machining processes (such as milling) in real world.
</h2>

**Signals** can be assumed as ***vibration in 3 axes*** (X, Y, Z), and we can see **y values** as ***surface roughness***, ***width of milled grooves***, or any other thing you come out with. It's time for your imagination.

Now, with signals and quality records gathered from machining processes, we want to **examine their relations**, as it is helpful for us to **understand the mechanisms underlying the machining process**. The major functionalities of the repo. can be divided into three topics, which are related to different .py files below:

<h1 align="center">
signal_processing.py
</h1>
<h2 align="center">
Processing the captured signals using time-domain, frequency-domain, or time-frequency-domain analyses.
</h2>
<h3 align="center">
I am gonna demonstrate how signal processing can be done with this file.
</h3>

First, we have to load our signal datasets & quality datasets. 

After that, **let's look at signals in one arbitrary run**.

```
signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset', first_signal_minus=False)
sample_rate = int(20000/10) # you have to know your signal sampling rate before analyzing signals.
time_runs = sigpro.pick_one_signal(signals_runs, signal_idx=0)
run_idx_demo = 10 # the designate index of process run, this number can be changed at will.
run_signals = signals_runs[run_idx_demo] # run_signals.shape = (4, 10296)
```

Importing signal_processing.py and see what signals does the run have.
```
import signal_processing as sigpro
import signal_plotting as sigplot
sigplot.draw_signals(run_signals[1:], run_signals[0], legend_lst=['X', 'Y', 'Z'], color_lst=['royalblue', 'peru', 'seagreen'], title='All vibration signals')
# run_signals[0] are time stamps, while run_signals[1:] are all 3 signals of the run
```
![Signals](image/signals_of_demo_run.png) 

As you can see, all 3 signals are plotted. 

***sigplot.draw_signals()*** doesn't care how many signals are in the run_signals. As long as **first row** stands for **time** and **the rest rows** are **signals**, every signal can be shown with this function.

You can designate the **names of signals** and the **presented color of signals** in form of **list**.

Then use those list as **arguments** for ***legend_lst*** and ***color_lst*** respectively.

Now, **if you want to shrink the size of signals, sigpro.signal_resize() can help**.

Let's specify that we want to see **the last signal**, and we want to see the **resized version** of the last signal of the 10th run.

```
siganl_idx_demo = 3
signal_runs = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo)
signals_resize, time_resize = sigpro.signal_resize(signal_runs, time_runs, final_length=5000) # shrink it to length=5000
sigplot.draw_signal(signal_runs[run_idx_demo], time_runs[run_idx_demo], color_='royalblue', title='OG Signal')
sigplot.draw_signal(signals_resize[run_idx_demo], time_resize[run_idx_demo], color_='seagreen', title='Resized Signal')
```
![OG Signals](image/Fsignal_resized_og.png) 
![Resized Signals](image/Fsignal_resized_resampled.png) 

This function use **resample()** function from **scipy.signal** ([Reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html "link" )), which exploit *FFT transformations*. You can see there are some differences between two signals, but the overall trends are very similar.


Speaking of *FFT*, **let's look at the frequency spectrum of this signal**.

```
sig_fft_runs = sigpro.get_frequency_spectra(signal_runs, sample_rate)
band_demo = sig_fft_runs[run_idx_demo][0]
spectrum_demo = sig_fft_runs[run_idx_demo][1]
sigplot.frequency_spectrum(band_demo, spectrum_demo)
```

![FFT](image/freq_spectrum.png) 

Frequencies around **120** & **1 Hz** are very prominent, which can be justified because ***sig3*** in ***waveMaker.py*** are constructed by signals components with those frequencies.

Now that we have the frequency specttrum, let's **band-pass it to change the signal**, shall we?

I am gonna **keep** the information within **frequency range 50~600**, and discard the rest

```
signals_filtered = sigpro.freq_pass(signal_runs, order_=2, assigned_freq=[50, 600], btype='bandpass', fs=sample_rate)
sig_fft_runs_filtered = sigpro.get_frequency_spectra(signals_filtered, sample_rate)
sigplot.draw_signal(signals_filtered [run_idx_demo], time_runs[run_idx_demo], title='Filtered')
band_demo = sig_fft_runs_filtered[run_idx_demo][0]
spectrum_demo = sig_fft_runs_filtered[run_idx_demo][1]
sigplot.frequency_spectrum(band_demo, spectrum_demo, title='Filtered')
```
![Filtered Signals](image/band_pass.png) 
![Frequency Spectrum of Filtered Signals](image/band_pass_spectrum.png) 

As you can see, **no more low-frequency trending** inside the signal, and the frequency spectrum contains **120 Hz as the main component**.

I use ***butterworth filter*** (from ***scipy.signal***) to do the band-passing, **arguments setting** of ***order_*** and ***btype*** can be change based on [scipy's website](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html "link" )

In some case, it's helpful to **inspect the envelopes of one signal**, as the **amplitude modulation** phenomenon can be dissected.

I find signal's **upper/lower envelopes** based on **local extrema**. But before doing that, I **smooth** the signal with a ***deformable Gaussian filter*** in advance. Such a process can **decrease the impact of noise** within the signal.

The Gaussian filter is exploited base on ***gaussian_filter1d()*** from ***scipy.ndimage*** ([Reference Here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html "link" ))

```
envelopes_up_runs, envelopes_low_runs = sigpro.get_envelope_lst(signal_runs, time_runs, gau_sig=10, gau_rad=20, w_size=30)
# arguments gau_sig, gau_rad, and w_size decide the shape of the Gaussian filter
sigplot.plot_envelope(signal_runs[run_idx_demo], time_runs[run_idx_demo], envelopes_up_runs[run_idx_demo], envelopes_low_runs[run_idx_demo])
```
![Envelopes](image/envelope.png) 


<h1 align="center">
featureExtraction.py & autoencoder.py
</h1>
<h2 align="center">
Extracting features from the signals.
</h2>

<h1 align="center">
cross-validation.py, correlation_analysis.py, and any .py with 'classPSO' as prefix
</h1>
<h2 align="center">
Analyzing the relations between extracted features and qualities, or constructing machine-learning models to predict the qualities based on input signals.
</h2>
