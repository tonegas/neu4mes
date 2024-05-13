import time
from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

#Create the motor trasmission
gear = Input('gear', values=[1,2,3,4,5,6,7,8])
engine = Input('engine')
motor_force = LocalModel(engine.tw(1), gear)

#Create the concept of the slope
altitude = Input('altitude')
gravity_force = Linear(altitude.tw([1,-1], offset = 0))

# Create the brake force contribution
brake = Input('brake')
brake_force = -Relu(Linear(brake.tw(1)))

#Create the areodinamic drag contribution
velocity = Input('velocity')
drag_force = Linear(velocity^2)
#Longitudinal acceleration
long_acc = Input('acceleration')

#Definition of the next acceleration predict 
long_acc_estimator = Output(long_acc.z(-1), motor_force+drag_force+gravity_force+brake_force)

# Add the neural model to the neu4mes structure and neuralization of the model
mymodel = Neu4mes(verbose = True, visualizer = StandardVisualizer())
mymodel.addModel(long_acc_estimator)
mymodel.neuralizeModel(0.05)

# Data load
data_struct = ['velocity','engine','brake','gear','travel','altitude','acc','velKal','acceleration']
data_folder = './examples/datasets/vehicle_data/'
mymodel.loadData(data_struct, folder = data_folder, skiplines = 1)

# Neural network train
start = time.time()
mymodel.trainModel(test_percentage = 30, show_results=True)
end = time.time()
print(end - start)