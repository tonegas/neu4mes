from neu4mes import *

input1 = Input('in1')
input2 = Input('in2')
output = Input('out')
rel1 = Linear(input1.tw(0.05))
rel2 = Linear(input1.tw(0.01))
rel3 = Linear(input2.tw(0.05))
rel4 = Linear(input2.tw([0.02,-0.02]))
fun = Output(output.z(-1),rel1+rel2+rel3+rel4)


test = Neu4mes(verbose=True)
test.addModel(fun)
test.neuralizeModel(0.01)
print(test.inputs)


# data_struct = ['x1','y1','x2','y2','','A1x','A1y','B1x','B1y','','A2x','A2y','B2x','out','','x3','in1','in2','time','']
# data_folder = './tests/data/'
# test.loadData(data_struct, folder = data_folder, skiplines = 4)

print(test.inout_asarray)