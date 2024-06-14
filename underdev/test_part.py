from neu4mes import *

in1 = Input('in1')

rel1 = Fir(in1.sw(1))
rel2 = Fir(in1.sw([-2,2]))
rel3 = Fir(in1.sw([-5,-2],offset=-3))
rel4 = Fir(SamplePart(in1.sw([-5,-2],offset=-3),-3,-2))
rel5 = Fir(in1.tw(1))
rel6 = Fir(in1.tw([-2,2]))
rel7 = Fir(in1.tw([-5,-2],offset=-3))
rel8 = Fir(TimePart(in1.tw([-5,-2],offset=-3),-3,-2))
out = Output('out', rel1+rel2+rel3+rel4+rel5+rel6+rel7+rel8)

# input1 = Input('in1')
# output = Input('out')
# rel1 = Linear(input1.tw(0.05))
# fun = Output(output.z(-1),rel1)

test = Neu4mes()
test.addModel(out)
test.neuralizeModel(0.5)
