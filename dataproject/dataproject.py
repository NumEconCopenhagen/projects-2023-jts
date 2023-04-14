import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns

def calculate_returns(data):
    # calculates monthly returns using the pct_change() function
    data_r = data.pct_change()

    # calculates cumulative returns using the cumprod() function
    data_cr = (1 + data_r).cumprod()
    
    # returns monthly and cumulative returns
    return data_r, data_cr


def calculate_portfolio_returns(data, weights):
    # calculate weighted returns
    weighted_r = data * weights

    # calculate portfolio returns
    port_r = weighted_r.sum(axis=1)

    # calculate cumulative return
    cum_port_r = (1 + port_r).cumprod()

    # return portfolio returns and cumulative returns
    return port_r, cum_port_r


def cum_ret_plot(data, stock, ref, fig = 1, ax_data=None):
    # set ax_data to pd.dataframe for a noninteractive comparison of two stocks
    if ax_data is None:
        ax_data = data

    # first plot of y using ax_data = data for interactive
    ax = ax_data.plot(y=stock) 
    # second plot of the reference 'index', contains the positional arg: ax = ax
    data.plot(y=ref,
              # only for noninteractive comparison otherwise ax = data
              ax=ax,
              # set title and ylabel
              title = f'Figure {fig}: Cumulative Return of {stock} compared to {ref}',
              ylabel = 'Cumulative Return')
    # display plot
    plt.show()


def plot_scatter_with_labels(ax, x, y, labels, title, xlabel, ylabel):
    # create the scatter plot with a regression line
    sns.regplot(x=x, y=y, scatter=False, ax=ax)
    ax.scatter(x, y)

    # add a list of text labels for each data point using ax.text and a for loop
    texts = [ax.text(x_pos, y_pos, f'{lab}', fontsize=8, ha='center') for (x_pos, y_pos, lab) in zip(x, y, labels)]

    # adjust the text labels for each data point automatically
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5), ax=ax)

    # set title and labels
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)


def normalize_column(col):
    # set col_sum to the sum of the column
    col_sum = col.sum()
    # for each value in column divide by col_sum
    norm_col = col / col_sum
    # returns normalized column
    return norm_col