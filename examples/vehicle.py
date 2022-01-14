from neu4mes import *
from pprint import pp, pprint

#Create the motor trasmission
gear = Input('gear', values=[1,2,3,4,5,6,7,8])
engine = Input('engine')
motor_force = LocalModel(engine.tw(1), gear)

#Create the concept of the slope
altitude = Input('altitude')
gravity_force = Linear(altitude.tw([5,-5], offset = 0))

# Create the brake force contribution
# Total of 25 samples
brake = Input('brake')
brake_force = -Relu(Linear(brake.tw(1.25)))

#Create the areodinamic drag contribution
velocity = Input('velocity')
drag_force = Linear(velocity^2)

#The state is a well established variable, so if it used inside two different outputs
#will not be duplicated.
# omega = State('omega',brake_force+drag_force)
#    'States':{
#        'omega':{}
#    },
#   'Relations:{
#       'omega':{
#           'Linear':[brake_force,drag_force]
#       }
#   }
#The State can be added in the back propagation schema is needed.
# omega = Input('omega') 
# omega_estimator = State('omega', brake_force+drag_force, out = omega.z(-1))
    # 'States':{
    #     'omega':{}
    # },
    # 'Outputs':{
    #     'omega':{}
    # },
# lat_acc = Input('lat_acc')
# lat_acc_estimator = Output(lat_acc.z(1), omega+drag_force+gravity_force+motor_force)

#Longitudinal acceleration
long_acc = Input('acceleration')

#Definition of the next acceleration predict 
long_acc_estimator = Output(long_acc.z(-1), motor_force+drag_force+gravity_force+brake_force)

mymodel = Neu4mes(verbose=True)
mymodel.addModel(long_acc_estimator)
mymodel.neuralizeModel(0.05)

data_struct = ['velocity','engine','brake','gear','travel','altitude','acc','velKal','acceleration']
data_folder = './vehicle_data/'
mymodel.loadData(data_struct, folder = data_folder, skiplines = 1)
mymodel.trainModel(test_percentage = 30, show_results=True)

#definisco che long_acc è l'integrale di velocity
# long_acc.s(-1) = velocity 
#sono uguali 
# velocity.s(1) = long_acc
#così dico che voglio mettere degli stati per fare il training ricorrente
# mymodel.trainModel(validation_percentage = 30, states = [long_acc_estimator])

#data_struct = ['time','velocity','accleration','engine','gear','altitude','brake']
#data_folder = './data/vehicle_data/'
#mymodel.loadData(data_struct, folder = data_folder)
#mymodel.trainModel(validation_percentage = 30)