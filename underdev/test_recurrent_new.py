import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *

example = 4

if example == 1:
    print('#### EXAMPLE 1 - Call with close_loop ####')
    ## la memoria non viene conservata tra predict successive
    x = Input('x') 
    F = Input('F')
    x_out = Fir(x.tw(0.5))+F.last()
    out = Output('out',x_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)

    mass_spring_damper.neuralizeModel(0.1)

    ## Prediction di 1 solo sample con F inizializzata a Zero
    mass_spring_damper(inputs={'x':[1,2,3,4,5]}, close_loop={'F':'out'})

    ## Prediction di 5 solo sample con F inizializzata a Zero la prima volta
    #  e poi viene inizializzata con out dalla seconda in avanti
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9]}, close_loop={'F':'out'})

    ## Prediction di 1 solo sample con F inizializzata a 1
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1]}, close_loop={'F':'out'})

    ## Prediction di 5 solo sample con F inizializzata a 1 la prima volta
    #  e poi viene inizializzata con out dalla seconda in avanti
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1]}, close_loop={'F':'out'})

    ## Prediction di 5 solo sample con F inizializzata le prime 3 volte e poi in close_loop
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9], 'F':[1,2,3]}, close_loop={'F':'out'})

if example == 2:
    print('#### EXAMPLE 2 - Call with States variables ####')
    ## la memoria viene conservata tra predict successive
    x = Input('x') 
    F_state = State('F')
    y_state = State('y')
    z_state = State('z')
    x_out = Fir(x.tw(0.5))+F_state.last()+y_state.last()+z_state.last()
    x_out.update(F_state)
    x_out.update(y_state)
    x_out.update(z_state)
    out = Output('out',x_out)

    mass_spring_damper = Neu4mes()
    mass_spring_damper.addModel(out)

    mass_spring_damper.neuralizeModel(0.1)

    ## Prediction di 1 solo sample con variabili di stato non inizializzate
    mass_spring_damper(inputs={'x':[1,2,3,4,5]})

    ## Prediction di 1 solo sample con F inizializzata ad 1 e le altre non inizializzate
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1]})

    ## Prediction di 1 solo sample con tutte le variabili inizializzate
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1], 'y':[2], 'z':[3]})

    ## Prediction di 1 solo sample con tutte le variabili inizializzate a zero
    mass_spring_damper(inputs={'x':[1,2,3,4,5]}, inizialize_state=True)

    ## Prediction di 1 solo sample con tutte le variabili inizializzate (i valori extra di inizializzazione vengono ignorati)
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1,2,3], 'y':[2,3], 'z':[3]})

    ## Prediction di 5 sample con variabili di stato non inizializzate
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7,8,9]})

    ## Prediction di 5 sample con tutte le variabili inizializzate
    mass_spring_damper(inputs={'x':[1,2,3,4,5,6,7], 'F':[1,2,3], 'y':[2,3], 'z':[3]})
    ## equal to 
    mass_spring_damper(inputs={'x':[1,2,3,4,5], 'F':[1], 'y':[2], 'z':[3]})
    mass_spring_damper(inputs={'x':[2,3,4,5,6], 'F':[2], 'y':[3]})
    mass_spring_damper(inputs={'x':[3,4,5,6,7], 'F':[3]})
