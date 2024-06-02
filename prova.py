# import multiprocessing as mp
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# import time
# import signal
# import sys
#
#
# def plot_graph(data_queue, stop_event):
#     def signal_handler(sig, frame):
#         print('Ctrl+C pressed plot')
#         stop_event.set()
#         plt.close('all')
#
#     signal.signal(signal.SIGINT, signal_handler)
#
#     plt.ion()
#     fig, ax = plt.subplots()
#     xdata, ydata = [], []
#     ln, = ax.plot([], [], 'ro')
#
#     def init():
#         ax.set_xlim(0, 10)
#         ax.set_ylim(0, 10)
#         return ln,
#
#     def update(frame):
#         while not data_queue.empty():
#             data = data_queue.get()
#             xdata.append(data[0])
#             ydata.append(data[1])
#             ln.set_data(xdata, ydata)
#         return ln,
#
#     def handle_close(evt):
#         stop_event.set()
#
#     plt.gcf().canvas.mpl_connect('close_event', handle_close)
#     ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50, save_count=50)
#     plt.show(block=False)
#
#     while not stop_event.is_set():
#         plt.pause(0.1)
#
#
# def update_data(data_queue, stop_event):
#     time.sleep(1)
#     for i in range(100):
#         #if stop_event.is_set():
#         #    break
#         x = i * 0.1
#         y = np.sin(x)
#         data_queue.put((x, y))
#         time.sleep(0.1)
#
# print('GAS')
# if __name__ == '__main__':
#     mp.set_start_method('spawn')  # Ensure the 'spawn' start method for compatibility
#     data_queue = mp.Queue()
#     stop_event = mp.Event()
#
#     def signal_handler(sig, frame):
#         print('Ctrl+C pressed __main__')
#         stop_event.set()
#         plot_process.join()
#         sys.exit()
#
#     signal.signal(signal.SIGINT, signal_handler)
#
#
#     plot_process = mp.Process(target=plot_graph, args=(data_queue, stop_event))
#     plot_process.start()
#
#
#     # try:
#     update_data(data_queue, stop_event)
#     # except KeyboardInterrupt:
#     #     stop_event.set()
#     # finally:
#     #     #stop_event.set()
#     plot_process.join()
#     #sys.exit()
# user_script.py

import subprocess
import time
import random
import json

# Path to the data visualizer script
visualizer_script = 'multiprov.py'

# Start the data visualizer process
process = subprocess.Popen(['python', visualizer_script], stdin=subprocess.PIPE, text=True)

try:
    while True:
        # Generate random data
        data = {"a": random.random(), "b": random.random()}
        try:
            # Send data to the visualizer process
            process.stdin.write(f"{json.dumps(data)}\n")
            process.stdin.flush()
        except BrokenPipeError:
            print("The visualizer process has been closed.")
            break
        # Wait for a short period before generating the next data point
        #time.sleep(0.001)
except KeyboardInterrupt:
    # Terminate the visualizer process gracefully
    process.terminate()
