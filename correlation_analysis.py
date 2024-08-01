import numpy as np
from matplotlib import pyplot as plt
# from sklearn.metrics import r2_score


def plot_scatter(variable_a, variable_b, corr_, title_='Pearson Correlation', content_=['Variable 1', 'Variable2']):
    plt.figure(figsize=(12, 9))
    plt.plot(variable_a, variable_b, 'o', color='purple', lw=5)
    topValue = (max(variable_a) if max(variable_a) > max(variable_b) else max(variable_b))
    topValue = topValue * 1.1 if topValue > 0 else topValue * 0.9
    bottomValue = (min(variable_a) if min(variable_a) < min(variable_b) else min(variable_b))
    bottomValue = bottomValue * 0.9 if topValue > 0 else topValue * 1.1
    plt.xlabel(f'{content_[0]}', fontsize=24)
    plt.ylabel(f'{content_[1]}', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title(f"{title_} \n Correlation={corr_:.2f}"
              , fontsize=26)
    plt.grid()


def get_corr_value_2variables(variable1, variable2, isPlot=True, title_='Pearson Correlation', content_=['Variable 1', 'Variable2']):
    corr = np.corrcoef(variable1, variable2, rowvar=True)[0][1]
    if isPlot:
        plot_scatter(variable1, variable2, corr, title_, content_)
    return corr

def features_vs_quality(x_, y_):
    x_ = x_.T # shape: (amount of features, amount of samples)
    y_ = y_.T # shape: (amount of quality, amount of samples)

    corr = np.zeros((y_.shape[0], x_.shape[0]))
    for feature_idx, feature in enumerate(x_): # correlation between 2 variables: feature & quality
        for y_idx, y_target in enumerate(y_):
            R = get_corr_value_2variables(feature, y_target, isPlot=False)
            corr[y_idx, feature_idx] = R
    return corr

def corr_filter(x_, corr_matrix_, threshold):
    target_idx = np.where(np.abs(corr_matrix_) >= threshold)[0]
    new_x = x_[:, target_idx]
    return target_idx, new_x

def plot_corr_distribution(corr_array, amount):
    feature_idx = np.arange(1, corr_array.shape[0]+1, 1)
    new_array = np.concatenate((feature_idx.reshape(-1, 1), np.abs(corr_array.reshape(-1, 1))), axis=1)
    new_array = new_array[new_array[:, 1].argsort()[::-1]]
    chosen_feature = []
    for feature_num in new_array[:amount, 0]:
        chosen_feature.append('{0:.0f}'.format(feature_num))
    plt.figure(figsize=(8, 4))
    plt.rcParams.update({'font.size': 18})
    plt.bar(np.array(chosen_feature), new_array[:amount, 1], color = 'purple')
    plt.grid()
    plt.ylabel('| correlation |', fontsize=26)
    plt.ylim(0, 1)
    plt.xlabel('Features', fontsize=28)
    plt.title('| Feature Correlation |', fontsize=30)
    return new_array

def plot_correlation_matrix(matrix):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value_percentage = (matrix[i, j] - np.amin(matrix)) / (np.amax(matrix) - np.amin(matrix)) 
            if value_percentage < 0.3:
                text_color = 'cyan'
            elif value_percentage >= 0.3 and value_percentage < 0.7:
                text_color = 'gold'
            else:
                text_color = 'darkred'
                
            text = ax.text(j, i, '{0:.3f}'.format(matrix[i, j]),
                           ha="center", va="center", color=text_color)
    ax.set_xticks(np.arange(0, matrix.shape[1], 1))   
    ax.set_yticks(np.arange(0, matrix.shape[0], 1))   
    ax.set_xlabel('Features')
    ax.set_ylabel('Y')
    ax.set_title('Pearson Correlation Matrix', fontsize=14)
    fig.tight_layout()


