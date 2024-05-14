from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

x = Input('x')
F = Input('F')

# Example 1
# Create a parameter k of dimension 3
k = Parameter('k', dimensions=3, tw=4)
fir1 = Fir(3, parameter=k)
fir2 = Fir(3, parameter=k)
out = Output(x.z(-1), fir1(x.tw(4))+fir2(F.tw(4)))
#

# Example 2
# Create
g = Parameter('g', dimensions=3 )
t = Parameter('k', dimensions=3 )
def fun(x, k, t):
    return x*k*t
p = ParamFun(fun, output_dimension=3, parameters=[k,t])
out = Output(x.z(-1), p(x.tw(1)))
#


mass_spring_damper = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
mass_spring_damper.addModel(out)
mass_spring_damper.neuralizeModel(0.05)

data_struct = ['time','x','x_s','F']
#data_folder = './datasets/mass-spring-damper/data/'
data_folder = os.path.join( 'datasets', 'mass-spring-damper', 'data')
mass_spring_damper.loadData(data_struct, folder = data_folder)

mass_spring_damper.trainModel(test_percentage = 10, show_results = True)