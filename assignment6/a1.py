import pandas as pd
import numpy as np

def equal_width_binning(data_column, num_bins=4):
    min_val = data_column.min()
    max_val = data_column.max()
    bin_width = (max_val - min_val) / num_bins
    bins = np.arange(min_val, max_val + 1e-9, bin_width)
    labels = [f'Bin {i+1}' for i in range(num_bins)]
    binned_data =