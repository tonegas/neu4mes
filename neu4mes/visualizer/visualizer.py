from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.CRITICAL, LOG_LEVEL))

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"
COLOR_BOLD_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

def color(msg, color_val = GREEN, bold = False):
    if bold:
        return COLOR_BOLD_SEQ % (30 + color_val) + msg + RESET_SEQ
    return COLOR_SEQ % (30 + color_val) + msg + RESET_SEQ

class Visualizer():
    def __init__(self):
        pass

    def set_n4m(self, n4m):
        self.n4m = n4m

    def warning(self, msg):
        print(color(msg, YELLOW))

    def showModel(self):
        pass

    def showMinimizeError(self,variable_name):
        pass

    def showModelInputWindow(self):
        pass

    def showModelRelationSamples(self):
        pass

    def showBuiltModel(self):
        pass

    def showDataset(self):
        pass

    def showTraining(self, epoch, train_losses, test_losses):
        pass

    def showTrainingTime(self, time):
        pass

    def showResults(self):
        pass