import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Define feature engineering function
def engineer_features(data):
    # Select top k features using ANOVA F-value
    selector = SelectKBest(f_classif, k=5)
    data_selected = selector.fit_transform(data.drop('Target', axis=1), data['Target'])

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_selected)

    return pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

# Define feature extraction function
def extract_features(data):
    # Extract Fourier transform features
    from scipy.fftpack import fft
    data_fft = fft(data['Time Series'])

    # Extract wavelet transform features
    from pywt import dwt
    data_dwt = dwt(data['Time Series'], 'haar')

    return pd.concat([pd.DataFrame(data_fft, columns=['FFT']), pd.DataFrame(data_dwt, columns=['DWT'])], axis=1)
