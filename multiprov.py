import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import json

# Buffer to hold the data points
data_buffer = deque(maxlen=2000)

def update_graph(frame):
    # Read data from stdin
    line = sys.stdin.readline().strip()
    if line:
        try:
            # Convert to float and append to buffer
            data_point = json.loads(line)
            key = list(data_point['train_losses'].keys())[0]
            data_buffer.append(data_point['train_losses'][key])
        except ValueError:
            pass

    # Clear the current plot
    plt.clf()
    # Plot data
    plt.plot(data_buffer)
    # Set plot limits
    plt.ylim(-1, 1)

# Set up the plot
fig, ax = plt.subplots()

# Use FuncAnimation to update the plot dynamically
ani = animation.FuncAnimation(fig, update_graph, interval=10, save_count=20)

# Show the plot
plt.show()
