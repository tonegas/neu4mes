import sys, json

import matplotlib.pyplot as plt

from mplplots import plots

line = sys.stdin.readline().strip()
key, A, B, sample_time = None, None, None, None
if line:
    try:
        # Convert to float and append to buffer
        data_point = json.loads(line)
        name_data = data_point['name_data']
        key = data_point['key']
        A = data_point['prediction_A']
        B = data_point['prediction_B']
        sample_time = data_point['sample_time']
    except ValueError:
        pass

fig, ax = plt.subplots()
ax.cla()
plots.plot_results(ax, name_data, key, A, B, sample_time)
plt.show()
