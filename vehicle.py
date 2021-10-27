from Neu4mes import Neu4mes, Input, Linear

#Vehicle example
model_def = {
    'SampleTime':0.05,
    'Input':{
        'gear':{
            'Distrete':[1,2,3,4,5,6,7,8]
        },
        'engine':{},
        'brake':{},
        'altitude':{},
        'velocity':{}
    },
    'State':{
        'omega':{}
    },
    'Output':{
        'acceleration':{}
    },
    'Relations':{
        'gravity_force':{
            'Linear':[('altitide',1)]
        },
        'motor_force':{
            'LocalModel':[('engine',1.25),'gear']
        },
        'lin_brake':{
            'Linear':[('brake',1.25)],
        },
        'relu_brake':{
            'Relu':['lin_brake'],
        },
        'brake_force':{
            'Minus':['relu_brake'],
        },
        'omega':{
            'Prova':['velocity']
        },
        'acceleration': {
            'Sum':['gravity_force','motor_force','brake_force'],
            'Square':['velocity']
        }
    }
}
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

#The state is a well established variable, so if it used inside two different outputs
#will not be duplicated.
# omega = State('omega',brake_force+drag_force)
#     'State':{
#        'omega':{}
#    },
#The State can be added in the back propagation schema is needed.
# omega = Input('omega') 
# omega_estimator = State('omega', brake_force+drag_force, out = omega.z(1))
    # 'State':{
    #     'omega':{}
    # },
    # 'Output':{
    #     'omega':{}
    # },
# lat_acc = Input('lat_acc')
# lat_acc_estimator = Output(lat_acc.z(1), omega+drag_force+gravity_force+motor_force)

#Longitudinal acceleration
long_acc = Input('accleration',velocity.s(1))


#Definition of the next acceleration predict 
long_acc_estimator = Output(long_acc.z(1), brake_force+drag_force+gravity_force+motor_force)


mymodel = Neu4mes()
mymodel.modelDefinition(long_acc_estimator)
mymodel.neuralizeModel()

data_struct = ['time','velocity','accleration','engine','gear','altitude','brake']
data_folder = './data/vehicle_data/'
mymodel.loadData(data_struct, folder = data_folder)
mymodel.trainModel(validation_percentage = 30)