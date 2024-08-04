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

[Detail of signal_processing.py](signal_processing.md "link" )
<br>

<h1 align="center">
featureExtraction.py & autoencoder.py
</h1>
<h2 align="center">
Extracting features from the signals.
</h2>

[Detail of featureExtraction.py](featureExtraction.md "link" )

[Detail of autoencoder.py](autoencoder.md "link" )
</br>

<br>
<h1 align="center">
cross-validation.py, correlation_analysis.py, and any .py with 'classPSO' as prefix
</h1>
<h2 align="center">
Constructing machine-learning models to predict the qualities based on input signals.
</h2>


[Detail of cross-validation.py](cross-validation.md "link" )

[Detail of PSO files](PSO.md "link" )

</br>
