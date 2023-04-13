import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def norm_col(col):
    col_sum = col.sum()
    norm_col = col / col_sum
    return norm_col


def cum_ret_plot(data, stock, fig = 1, ax_data=None):
    if ax_data is None:
        ax_data = data

    # first plot of y using ax_data
    ax = ax_data.plot(y=stock) 
    # second plot of OMXC25 which contains the positional arg: ax = ax
    data.plot(y='^OMXC25',
              ax=ax,
              title = f'Figure {fig}: Cumulative Return of {stock} compared to OMXC25',
              ylabel = 'Cumulative Return')
    # display plot
    plt.show()