from Neu4mes import Neu4mes

#Vehicle example
# model_def = {
#     'SampleTime':0.05,
#     'Input':{
#         'gear':{
#             'Properties':['distrete']
#         },
#         'engine':{},
#         'brake':{},
#         'altitude':{},
#         'velocity':{}
#     },
#     'Output':{
#         'acceleration':{}
#     },
#     'Relations':{
#         'slope':{
#             'Linear':[('altitide',1)]
#         },
#         'torque':{
#             'LocalModel':[('engine',1.25),'gear']
#         },
#         'acceleration': {
#             'LinearNegative':[('brake',1.25)],
#             'Linear':['slope','torque'],
#             'Square':['velocity']
#         }
#     }
# }
#Create the motor trasmission
gear = DiscreteInput('gear', [1,2,3,4,5,6,7,8])
engine = Input('engine')
motor_force = LocalModel(engine.tw(2), gear)

#Create the concept of the slope
altitude = Input('altitude')
gravity_force = Linear(altitude.tw(2))

#Create the brake force contribution
brake = Input('brake')
brake_force = -Relu(Linear(brake.tw(1.25)))

#Create the areodinamic drag contribution
velocity = Input('velocity')
drag_force = Linear(velocity^2)

#Longitudinal acceleration
long_acc = Input('accleration',velocity.s(1))

#Definition of the next acceleration predict 
long_acc_estimator = Output('accleration', long_acc.z(1), brake_force+drag_force+gravity_force+motor_force)

mymodel = Neu4mes()
mymodel.modelDefinition(long_acc_estimator)
mymodel.neuralizeModel()

data_struct = ['time','velocity','accleration','engine','gear','altitude','brake']
data_folder = './data/vehicle_data/'
mymodel.loadData(data_struct, folder = data_folder)
mymodel.trainModel(validation_percentage = 30)