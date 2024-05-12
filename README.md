# Data mining with time series
I originally made this repo. to deal with a cooperative education program I participated in during my study. This Python library contains .py files to perform signal processing, constructing machine learning algorithms, and plotting signal spectra. These files helped me a lot, and hopefully, they may be found useful in your projects as well.

In this repo., one data and a folder are examples of how to use the .py contained. For a better understanding, I am gonna make an imaginary scenario in which these example files originate.
## Imaging a machining process that can be monitored by working conditions such as vibrations, temperature, or any other signals. The goal is to predict the resultant quality of the process w/ the captured signals.
First, the example files are created, which are 'demo_y.csv' and folder 'demonstration_signal_dataset'. The former is the quality records of several machining processes, and the latter contains their corresponding signals (.csv) captured during processes. 'waveMaker.py' describes the details of determining signals and quality values.

With signals and quality records gathered from processes, we want to examine their relations, as it is helpful for us to understand the mechanisms underlying the machining process. The major functionalities of the repo. can be divided into three topics, which are related to different .py files below:
### signal_processing.py: Processing the captured signals using time-domain, frequency-domain, or time-frequency-domain analyses.
### featureExtraction.py & autoencoder.py: Extracting features from the signals.
### cross-validation.py, correlation_analysis.py, and any .py with 'classPSO' as prefix: Analyzing the relations between extracted features and qualities, then constructing machine-learning models to predict the qualities based on input signals.
