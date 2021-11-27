from neu4mes import *

# model_def = {
#     'SampleTime':0.05,
#     'Input':{
#         'x1':{
#             'Name':'Mass 1 position'
#         },
#         'F':{
#             'Name':'External force'
#         }
#     },
#     'Output':{
#         'x1__-z1':{     #con z indico il ritardo unitario di una variabile     
#             'Name':'Next mass 1 position'
#         }
#     },
#     'Relations':{
#         'x1__-z1':{
#             'Linear':[('x1',2),'F'],
#         }
#     }
# }

springDamper = Neu4mes()
x1 = Input('x1')
F = Input('F')
x1_z = Output(x1.z(-1), Linear(x1.tw(2))+Linear(F))
springDamper.addModel(x1_z)
springDamper.neuralizeModel(0.05)
data_struct = ['time','x1','x1_s','F']
data_folder = './data/data-linear-oscillator-a/'
springDamper.loadData(data_struct, folder = data_folder)
springDamper.trainModel(validation_percentage = 30)

