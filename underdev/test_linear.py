import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

x = Input('x', dimensions=10)
FullyConnected([3,'Tanh',10,'Tanh'])

Linear()(x.last())
