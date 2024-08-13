<br>
<h1 align="center">
cross-validation.py
</h1>
<h2 align="center">
Constructing machine-learning models to predict the qualities based on input signals.
</h2>

### Time to **predict the resultant quality** of machining process with ***working parameters*** or ***extracted signal features***.

**[Review of the imaginary machining scenario](README.md "link" )**

**[Review of the extraction of signal features](featureExtraciton.md "link" )**

```
import signal_processing as sigpro

signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset', first_signal_minus=False)
sample_rate = int(20000/10)
y = np.genfromtxt('demo_y.csv', delimiter=',')
siganl_idx_demo = 3
signal_runs = sigpro.pick_one_signal(signals_runs, signal_idx=siganl_idx_demo)
time_runs = sigpro.pick_one_signal(signals_runs, signal_idx=0)
```


```
import featureExtraction as feaext
import cross_validation as cv

features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
domain_energy = features_freq.domain_energy
cv_prepare = cv.cross_validate(domain_energy, y[:, y_idx_demo], qualityKind=f'Y{y_idx_demo}')
param_setting = {'eta':0.3, 'gamma':0.01, 'max_depth':6, 'subsample':0.8, 'lambda':50, 'random_state':75}
trained_model = cv_prepare.cross_validate_XGB(param_setting=param_setting)
cv_prepare.model_testing(trained_model, 'XGB')
```

![CV_history](image/cv_run1_xgb.png)
![CV](image/cv_xgb.png)
![CV_test](image/cv_xgb_test.png)

```
features_time = feaext.TimeFeatures(signal_runs,target_lst=['rms', 'kurtosis', 'skewness', 'variance', 'p2p'])
features = features_time.features_all_signals
cv_prepare = cv.cross_validate(features, y[:, y_idx_demo], qualityKind=f'Y{y_idx_demo}')
trained_model = cv_prepare.cross_validate_DNN(dense_coeff=10)
cv_prepare.model_testing(trained_model, 'DNN')
```
![CV_history](image/cv_run1_dnn.png)
![CV](image/cv_dnn.png)
![CV_test](image/cv_dnn_test.png)
