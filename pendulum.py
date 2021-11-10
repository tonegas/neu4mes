from Neu4mes import *
import tensorflow.keras.backend

# massamolla = Neu4mes()
# x = Input('x')
# F = Input('F')
# x_z = Output(x.z(-1), Linear(x.tw(2))+Linear(F))
# massamolla.addModel(x_z)
# massamolla.neuralizeModel(0.05)
# 
# data_struct = ['time','x','x_s','F']
# data_folder = './data/data-oscillator-linear/'
# massamolla.loadData(data_struct, folder = data_folder)
# 
# massamolla.trainModel(validation_percentage = 30)

pendolo = Neu4mes()

theta = Input('theta')
T     = Input('T')

lin_theta = Linear(theta.tw(1.5))

#sin = tensorflow.keras.backend.sin(theta)
sin_theta = LinearBias(Sin(theta.tw(1.5)))

torque = Linear(T)

new_theta = Output(theta.z(-1), lin_theta+sin_theta+torque)
new_theta2 = Output(T.z(-1), lin_theta+sin_theta+torque)

pendolo.addModel(new_theta)
#pendolo.addModel(new_theta2)

pendolo.neuralizeModel(0.05)

data_struct = ['time','theta','theta_s','cos_theta','sin_theta','T']
data_folder = './data/data-pendulum-b/'
pendolo.loadData(data_struct, folder = data_folder)

pendolo.trainModel(validation_percentage = 30, show_results = True)

#mymodel.addModel(model_def)
#mymodel.neuralizeModel()

# data = {
#     'time' : [[],[]]
#     'x1' : [[simulazione 1],[simulazione 2]]
#     'F'  : [[],[]]
# }
#data_struct = ['time','x1','x1_s','F']
#data_folder = './data/data-oscillator-linear/'
#mymodel.loadData(data_struct, folder = data_folder)
#mymodel.trainModel(validation_percentage = 30)
#mymodel.showResults()


#Examples:
#1. Vehicle model + control system
#2. State estimator for lateral velocity
#3. Signle/double Mass-spring-dumper
#4. Cart-Pole
#5. Pedestrian estrimator
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
#         'x1_z':{     #con z indico il ritardo unitario di una variabile     
#             'Name':'Next mass 1 position'
#         },
#         'x1_s':{    #con s indico la defivata di un segnale
#             'Name':'Velocity of mass 1'
#         }
#     },
#     'Relations':{
#         'x1_z':{
#             'Linear':[('x1',2),'F'],
#         },
#         'x1_s':{
#             'Linear':[('x1',3),'F'],
#         },
#     }
# }
# model_def = {
#     'SampleTime':0.05,
#     'Input':{
#         'x1':{
#             'Name':'Mass 1 position'
#         },
#         'x2':{
#             'Name':'Mass 2 position'
#         },
#         'F':{
#             'Name':'External force'
#         },
#         'T':{
#             'Name':'External tension'
#         }
#     },
#     'Output':{
#         'x1p':{
#             'Name':'Next mass 1 position'
#         },
#         'x2p':{
#             'Name':'Next mass 2 position'
#         }
#     },
#     'Relations':{
#         'x1p':{
#             'Linear':[('x1',2),'F'],
#         },
#         'x2p':{
#             'Linear':[('x2',2),('F',1),'T'],
#         },
#     }
# }

# model_def = {
#     'Input':{
#         'omega':{},
#         'u':{},
#         'ay':{},
#         'ax':{}
#     },
#     'Output':{
#         'v':{}
#     },
#     'Params':{
#         'gamma':{},
#         'beta':{}
#     },
#     'State':{
#         'local_u':{},
#         'loval_v':{}
#     },
#     'Relations':{
#         'local_v':{
#             'Function':{
#                'eq': lambda v, ay, u, omega: mymodel.gamma*v+mymodel.beta*mymodel.Ts*(ay-v*omega)
#             }
#         },
#         'local_u':{
#             'Function':{
#                'eq': lambda v, ay, u, omega: mymodel.gamma*v+mymodel.beta*mymodel.Ts*(ay-v*omega)
#             }
#         },
#         'v':{
#             'LocalModel':[('local_v'),'omega']
#         }
#     }
# }