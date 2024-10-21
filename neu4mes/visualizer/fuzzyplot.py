import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import json
import numpy as np

fig, ax = plt.subplots()
# Clear the current plot
plt.clf()
# Plot data
line = sys.stdin.readline().strip()
name, x, y = None, None, None
if line:
    try:
        # Convert to float and append to buffer
        data_point = json.loads(line)
        name = data_point['name']
        x = data_point['x']
        for key in data_point['y'].keys():
            y = data_point['y'][key]
            plt.plot(x, y, label=f'Channel {int(key)+1}', linewidth=2)
    except ValueError:
        exit()

plt.legend(loc='best')
plt.xlabel('Input')
plt.ylabel('Value')
plt.title(f'Function {name}')
plt.show()