import sys, json

import matplotlib.pyplot as plt
import numpy as np


# Plot data
line = sys.stdin.readline().strip()
name, x, x0, x1, params, output = None, None, None, None, None, None
if line:
    try:
        # Convert to float and append to buffer
        data_point = json.loads(line)
        name = data_point['name']
        if 'x1' in data_point.keys():
            x0 = data_point['x0']
            x1 = data_point['x1']
        else:
            x = data_point['x0']
        params = data_point['params']
        input_names = data_point['input_names']
        output = data_point['output']
    except ValueError:
        exit()

fig = plt.figure()
# Clear the current plot
plt.clf()
if 'x1' in data_point.keys():
    x0, x1 = np.meshgrid(x0, x1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x0, x1,np.array(output), cmap='viridis')
    ax.set_xlabel(input_names[0])
    ax.set_ylabel(input_names[1])
    ax.set_zlabel(f'{name} output')
    for ind in range(len(input_names) - 2):
        fig.text(0.01, 0.9 - 0.05 * ind, f"{input_names[ind + 2]} ={params[ind]}", fontsize=10, color='blue',
                 style='italic')

else:
    plt.plot(np.array(x), np.array(output),  linewidth=2)
    plt.xlabel(input_names[0])
    plt.ylabel(f'{name} output')
    for ind in range(len(input_names) - 1):
        fig.text(0.01, 0.9 - 0.05 * ind, f"{input_names[ind + 1]} ={params[ind]}", fontsize=10, color='blue',
                 style='italic')

plt.title(f'Function {name}')
plt.show()