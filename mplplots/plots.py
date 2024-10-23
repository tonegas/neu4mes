def plot_training(plt, title, epoch, data_train, data_val = None, last = None):
    # Clear the current plot
    plt.clf()
    # Plot data
    if last is not None:
        plt.title(f'{title} - epochs last {last}')
    else:
        plt.title(f'{title}')

    plt.plot([i + 1 for i in range(epoch + 1)], data_train, label='Train loss')
    if data_val:
        plt.plot([i + 1 for i in range(epoch + 1)], data_val, '-.', label='Validation loss')

    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # Set plot limits
    if data_val:
        min_val = min([min(data_val), min(data_train)])
        max_val = max([max(data_val), max(data_train)])
    else:
        min_val = min(data_train)
        max_val = max(data_train)
    plt.ylim(min_val - min_val / 10, max_val + max_val / 10)