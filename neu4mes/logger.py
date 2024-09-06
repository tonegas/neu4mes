import logging
import sys

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[%dm"
COLOR_BOLD_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

COLORS = {
    logging.DEBUG: MAGENTA,
    logging.INFO: WHITE,
    logging.WARNING: YELLOW,
    logging.CRITICAL: RED,
    logging.ERROR: RED
}

class JsonFormatter(logging.Formatter):
    FORMAT = "[%(levelname)s][%(name)s:%(filename)s:%(funcName)s:%(lineno)d] %(message)s" # + ""
    def __init__(self):
        logging.Formatter.__init__(self, self.FORMAT)

    def format(self, record):
        result = logging.Formatter.format(self, record)
        result = COLOR_SEQ % (30 + COLORS[record.levelno]) + result + RESET_SEQ
        return result


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    handler = []
    logger = []
    def __init__(self, name):
        logging.Logger.__init__(self, name)

        #file = logging.FileHandler('example.log')
        #color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        self.console = logging.StreamHandler(sys.stdout)
        color_formatter = JsonFormatter()

        self.console.setFormatter(color_formatter)
        #self.console.setLevel(logging.CRITICAL)

        #logging.getLogger().addHandler(self.console)
        self.addHandler(self.console)
        self.handler.append(self.console)
        self.logger.append(self)
        #self.addHandler(file)

        return

    def disable(self, *args, **kwargs):
        self._log(logging.INFO, COLOR_SEQ % (30 + BLUE) + (" Disable Log ").center(80, '=') + RESET_SEQ, args,  **kwargs)
        # Remove all existing log handlers
        for ind, logger in enumerate(self.logger):
             logger.removeHandler(self.handler[ind])

    def enable(self, *args, **kwargs):
        # Add all existing log handlers
        for ind, logger in enumerate(self.logger):
            logger.addHandler(self.handler[ind])
        self._log(logging.INFO, COLOR_SEQ % (30 + BLUE) + (" Enable Log ").center(80, '=') + RESET_SEQ, args, **kwargs)


logging.setLoggerClass(ColoredLogger)

