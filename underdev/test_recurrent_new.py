import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

example = 1
## TODO: sposta in test usando assert
if example == 1:
    print('#### EXAMPLE 1 - Call with close_loop ####')
    ## la memoria non viene conservata tra predict successive
    x = Input('x') 
    F = Input('F')
    p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
    x_out = Fir(parameter=p)(x.tw(0.5))+F.last()
    out = Output('out',x_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)

    mass_spring_damper.neuralizeModel(0.1)

    ## Prediction di 1 solo sample con F inizializzata a Zero
    print('#### Predict 1 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5]}, close_loop={'F':'out'})

    ## Prediction di 5 solo sample con F inizializzata a Zero la prima volta
    ## e poi viene inizializzata con out dalla seconda in avanti
    print('#### Predict 2 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9]}, close_loop={'F':'out'})

    ## Prediction di 1 solo sample con F inizializzata a 1
    print('#### Predict 3 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1]}, close_loop={'F':'out'})

    ## Prediction di 5 solo sample con F inizializzata a 1 la prima volta
    #  e poi viene inizializzata con out dalla seconda in avanti
    print('#### Predict 4 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1]}, close_loop={'F':'out'})

    ## Prediction di 5 solo sample con F inizializzata le prime 3 volte e poi in close_loop
    print('#### Predict 5 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3]}, close_loop={'F':'out'})

    ## Prediction di 5 solo sample con F inizializzata le prime 3 volte e poi in close_loop
    print('#### Predict 6 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1,2,3]}, close_loop={'F':'out'})

elif example == 2:
    print('#### EXAMPLE 2 - Call with States variables ####')
    ## la memoria viene conservata tra predict successive
    x = Input('x') 
    F_state = State('F')
    y_state = State('y')
    z_state = State('z')
    p = Parameter('p', tw=0.5, dimensions=1, values=[[1.0],[1.0],[1.0],[1.0],[1.0]])
    x_out = Fir(parameter=p)(x.tw(0.5))+F_state.last()+y_state.last()+z_state.last()
    x_out.update(F_state)
    x_out.update(y_state)
    x_out.update(z_state)
    out = Output('out',x_out)

    mass_spring_damper = Neu4mes(seed=42)
    mass_spring_damper.addModel(out)

    mass_spring_damper.neuralizeModel(0.1)

    ## Prediction di 1 solo sample con variabili di stato non inizializzate
    ## (in memoria avranno lo stato dell'ultima predict)
    print('#### Predict 1 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5]})

    ## Prediction di 5 sample con variabili di stato non inizializzate
    ## e poi aggiornate con il valore della relazione di update
    print('#### Predict 2 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9]})

    ## Prediction di 1 solo sample con variabili di stato inizializzate a zero
    print('#### Predict 3 ####')
    mass_spring_damper.clear_state()
    mass_spring_damper(inputs={'x':[1,2,3,4,5]})

    ## Prediction di 1 solo sample con F inizializzata ad 1 e le altre non inizializzate
    print('#### Predict 4 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1]})

    ## Prediction di 1 solo sample con tutte le variabili inizializzate
    print('#### Predict 5 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1], 'y':[2], 'z':[3]})

    ## Prediction di 5 sample con tutte le variabili inizializzate 
    ## tante volte quanto la lunghezza delle rispettive window
    print('#### Predict 6 ####')
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3], 'y':[2,3], 'z':[3]})
