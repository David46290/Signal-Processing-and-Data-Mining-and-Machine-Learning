<h1 align="center">
featureExtraction.py
</h1>
<h2 align="center">
Extracting features from the signals.
</h2>

```
signals_runs = sigpro.get_signals('.\\demonstration_signal_dataset', first_signal_minus=False)
    sample_rate = int(20000/10)
    y = np.genfromtxt('demo_y.csv', delimiter=',')
    time_runs = sigpro.pick_one_signal(signals_runs, signal_idx=0)
    run_idx_demo = 10
    siganl_idx_demo = 3
```



```
features_time = feaext.TimeFeatures(signal_runs, target_lst=['rms', 'kurtosis',  'skewness', 'variance', 'p2p'])
features = features_time.features_all_signals
features_name = features_time.feature_names
print(features.shape) # (50, 5), signals from 50 runs, and each one of them yields 5 features
print(features_name) # (50, 5)
```

```
features_freq = feaext.FreqFeatures(signal_runs, sample_rate, num_wanted_freq=3)
domain_fre = features_freq.domain_frequency
domain_energy = features_freq.domain_energy
domain_fre_name = features_freq.feature_name_freq
domain_energy_name = features_freq.feature_name_energy

print(domain_fre.shape) # (50, 3), signals from 50 runs, and each one of them give 3 features.
print(domain_fre_name.shape) # (50, 1), each signal from 50 runs give the feature called 'Top 3 Frequencies of Signal'.

print(domain_energy.shape) # (50, 3), signals from 50 runs, and each one of them give 3 features.
print(domain_energy_name.shape) # (50, 1), each signal from 50 runs give the feature called 'Top 3 Energies of Signal'.
```

```
y_idx_demo = 1
if plot_corr:
        feature_idx = 3
        corr.get_corr_value_2variables(features[:, feature_idx], y[:, y_idx_demo], title_='Pearson Correlation', content_=[f'{features_name[0, feature_idx]} of signal {siganl_idx_demo+1}', f'Y{y_idx_demo+1}'])
    
if plot_matrix:
    features_time_y_corr = corr.features_vs_quality(features, y)
    corr.plot_correlation_matrix(features_time_y_corr)
```