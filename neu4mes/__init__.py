# __version__ = '0.0.7'
# __version__ = '0.0.8' Preliminary version
# __version__ = '0.1.0' ERC version
__version__ = '0.3.0'   #Pytorch version

import sys
import time

major, minor = sys.version_info.major, sys.version_info.minor

if major < 3:
    sys.exit("Sorry, Python 2 is not supported. You need Python >= 3.6 for "+__package__+".")
elif minor < 6:
    sys.exit("Sorry, You need Python >= 3.6 for "+__package__+".")
else:
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>---- '+ __package__+' ----<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

import logging
LOG_LEVEL = logging.DEBUG

from neu4mes.neu4mes import Neu4mes

from neu4mes.relation import ToStream, Stream, NeuObj, MAIN_JSON
from neu4mes.input import Input, State, Connect, ClosedLoop
from neu4mes.output import Output

from neu4mes.optimizer import Optimizer, SGD, Adam

from neu4mes.activation import Relu, Tanh
from neu4mes.fir import Fir
from neu4mes.linear import Linear
from neu4mes.arithmetic import Add, Sum, Sub, Mul, Pow, Neg
from neu4mes.trigonometric import Sin, Cos, Tan
from neu4mes.parametricfunction import ParamFun
from neu4mes.fuzzify import Fuzzify
from neu4mes.part import TimePart, TimeSelect, SamplePart, SampleSelect, Part, Select, VectToSample
from neu4mes.localmodel import LocalModel
from neu4mes.parameter import Parameter, Constant
from neu4mes.timeoperation import Int
from neu4mes.logger import logging
from neu4mes.visualizer import Visualizer, TextVisualizer, MPLVisulizer
from neu4mes.initializer import init_negexp, init_lin, init_constant
from neu4mes.exporter import Exporter, StandardExporter

import os, os.path, logging
from pprint import pp, pprint
import numpy as np