import logging
from pprint import pformat
import sys

#
# class CustomFormatter(logging.Formatter):
#
#     grey = "\x1b[38;20m"
#     yellow = "\x1b[33;20m"
#     red = "\x1b[31;20m"
#     bold_red = "\x1b[31;1m"
#     reset = "\x1b[0m"
#     format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
#
#     FORMATS = {
#         logging.DEBUG: grey + format + reset,
#         logging.INFO: grey + format + reset,
#         logging.WARNING: yellow + format + reset,
#         logging.ERROR: red + format + reset,
#         logging.CRITICAL: bold_red + format + reset
#     }
#
#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

#The background is set with 40 plus the number of the color, and the foreground with 30

#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        record.msg = pformat(record.msg)
        return logging.Formatter.format(self, record)


class JsonFormatter(logging.Formatter):

    err_fmt  = "ERROR: %(msg)s"
    dbg_fmt  = "DBG: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(message)s"
    FORMAT = " [  %(name)s:%(filename)s:%(funcName)s:%(lineno)d  ] " # + ""
              # "[%(levelname)-18s]  "
              # "%(message)s "
              # "($BOLD%(filename)s"
              # "$RESET:%(lineno)d)"
              #)
    user_fmt = "[%(asctime)s][%(name)s][(%(filename)s:%(lineno)d)] %(message)s"

    def __init__(self, user):
        self.user = user
        if user:
            logging.Formatter.__init__(self, self.user_fmt)
        else:
            logging.Formatter.__init__(self, self.FORMAT)

    def format(self, record):
        if self.user:
            result = logging.Formatter.format(self, record)
        else:
            format_orig = self._fmt
            if record.levelno == logging.DEBUG:
                #record.name = record.name.center(21, ' ')
                result = logging.Formatter.format(self, record)
                result = result.center(80, "=")
                result = COLOR_SEQ % (30 + MAGENTA) + result + RESET_SEQ
                #record.levelname = COLOR_SEQ % (30 + BLUE) + record.levelname + RESET_SEQ
                #record.filename =  record.filename.center(15,' ')
                #record.lineno = str(record.lineno).center(4,' ')
                #record.msg = COLOR_SEQ % (30 + BLUE) + record.msg + RESET_SEQ
            elif record.levelno == logging.INFO:
                self._style._fmt = self.info_fmt
                record.msg = COLOR_SEQ % (30 + GREEN) + record.msg + RESET_SEQ
                result = logging.Formatter.format(self, record)
            self._style._fmt = format_orig
        return result


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    #FORMAT = "[$BOLD%(name)-20s$RESET][%(levelname)-18s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    #COLOR_FORMAT = formatter_message(FORMAT, True)
    handler = []
    logger = []
    def __init__(self, name):
        logging.Logger.__init__(self, name)

        #file = logging.FileHandler('example.log')
        #color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        self.console = logging.StreamHandler(sys.stdout)
        color_formatter = JsonFormatter(name.find('neu4mes') == -1)

        self.console.setFormatter(color_formatter)
        #self.console.setLevel(logging.CRITICAL)

        #logging.getLogger().addHandler(self.console)
        self.addHandler(self.console)
        self.handler.append(self.console)
        self.logger.append(self)
        #self.addHandler(file)

        return

    #def debug(self, msg = None, *args, **kwargs):
    #    if self.isEnabledFor(logging.DEBUG):
    #        self._log(logging.DEBUG, msg, args, **kwargs)

    def titlejson(self, title, msg, level=logging.INFO, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, title, args, **kwargs)
        if self.level <= level:
            self._log(logging.INFO, (' '+title+' ').center(80, '='), args, **kwargs)
            self._log(logging.INFO, pformat(msg), args, **kwargs)
            self._log(logging.INFO, '='.center(80, '='), args, **kwargs)

    def title(self, title, level=logging.INFO, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, title, args, **kwargs)
        if self.level <= level:
            self._log(logging.INFO, (' ' + title + ' ').center(80, '='), args, **kwargs)

    def line(self, level=logging.INFO, *args, **kwargs):
        if self.level <= level:
            self._log(logging.INFO, ("=").center(80, '='), args, **kwargs)

    def paramjson(self, name, value, dim=30, level=logging.INFO, *args, **kwargs):
        if self.level <= level:
            lines = pformat(value, width=80-dim).strip().splitlines()
            vai = ('\n'+(' '*dim)).join(x for x in lines)
            #pformat(value).strip().splitlines().rjust(40)
            self._log(logging.INFO, (name).ljust(dim) + vai, args, **kwargs)

    def param(self, name, value, dim=30, level=logging.INFO, *args, **kwargs):
        if self.level <= level:
            self._log(logging.INFO, (name).ljust(dim) + value, args, **kwargs)

    def string(self, string, level=logging.INFO, *args, **kwargs):
        if self.level <= level:
            self._log(logging.INFO, string, args, **kwargs)

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

