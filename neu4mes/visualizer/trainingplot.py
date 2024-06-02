import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import json
import signal

# Buffer to hold the data points
data_train = deque(maxlen=2000)
data_test = deque(maxlen=2000)
last = 1
title = 'Training'
epoch = 0

def update_graph(frame):
    global last, title, epoch
    if last > 0:
        # Read data from stdin
        line = sys.stdin.readline().strip()
        if line:
            try:
                # Convert to float and append to buffer
                data = json.loads(line)
                data_train.append(data['train_losses'])
                data_test.append(data['test_losses'])
                title = data['key']
                last = data['last']
                epoch = data['epoch']
            except ValueError:
                pass

        # Clear the current plot
        plt.clf()
        # Plot data
        plt.title(f'{title} - epochs last {last}')
        plt.plot([i+1 for i in range(epoch+1)],data_train, label='Train loss')
        plt.plot([i+1 for i in range(epoch+1)],data_test, '-.', label='Test loss')

        plt.yscale('log')
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # Set plot limits
        min_val = min([min(data_test), min(data_train)])
        max_val = max([max(data_train), max(data_train)])
        plt.ylim(min_val-min_val/10, max_val+max_val/10)
    else:
        pass

# Set up the plot
fig, ax = plt.subplots()

# Use FuncAnimation to update the plot dynamically
ani = animation.FuncAnimation(fig, update_graph, interval=10, save_count=20)

# Show the plot
plt.show(block=True)
