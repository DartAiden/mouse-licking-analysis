import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import math

def mean(arr):
    return sum(arr) / len(arr)
def cross_correlation(x, y):
    x_mean = mean(x)
    y_mean = mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_sq_diff = sum((a - x_mean) ** 2 for a in x)
    y_sq_diff = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    if denominator == 0:
        return 'N/A'
    else:
        correlation = numerator / denominator
        return correlation

open('.gitignore','w')

fig2,ax2 = plt.subplots(4,4, figsize = (60,16))

x = 0
y = 0

for a in os.listdir('final_data\complete_ts'):
    title = a.split('_')[0]
    a = os.path.join(('final_data\complete_ts'),a)
    df = pd.read_csv(a)
    ax = ax2[x,y]
    ax.plot(df['Times'],df['Lick_Signal'], label = "Lick frames", color = "#f5474d", linewidth = .25)
    ax.plot(df['Times'],df['Stim_Signal'], label = "Stimulation frames", color = "#74abfc", linewidth = .25)
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlim([0,df['Times'].iloc[-1]])
    cc = cross_correlation(df['Lick_Signal'],df['Stim_Signal'])
    ax.text(0.5, -0.37, f'Cross-correlation = {cc}', ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    x+=1
    if x ==4:
        x = 0
        y+=1
    if x == 3 and y == 3:
        ax.legend()
fig2.delaxes(ax2.flatten()[11])
fig2.delaxes(ax2.flatten()[15])
fig2.delaxes(ax2.flatten()[7])
plt.subplots_adjust(wspace=0.4, hspace=0.6)
handles, labels = ax2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.9, 0.1), fontsize=14)
plt.show()