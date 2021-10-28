import Relation
import Input
import tensorflow.keras.layers

class Linear(Relation.Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_lin'
            self.json['Relations'][self.name] = {
                'Linear':[(obj[0].name,obj[1])],
            }
        elif type(obj) is Input.Input:
            super().__init__(obj.json)
            self.name = obj.name+'_lin'
            self.json['Relations'][self.name] = {
                'Linear':[obj.name]
            }
        elif type(obj) is Relation.Relation:
            super().__init__(obj.json)
            self.name = obj.name+'_lin'
            self.json['Relations'][self.name] = {
                'Linear':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')
                  
    def setInput(self, model, relvalue):
        for el in relvalue:
            if type(el) is tuple:
                time_window = model.input_time_window.get(el[0])
                if time_window:
                    if model.input_time_window[el[0]] < el[1]:
                        model.input_time_window[el[0]] = el[1]
                else:
                    model.input_time_window[el[0]] = el[1]
            else: 
                model.input_time_window[el] = model.model_def['SampleTime']

    def createElem(self, model, relvalue, outel):
        for el in relvalue:
            if type(el) is tuple:
                if model.relations.get(el[0]) is None:
                    model.relations[el[0]+outel] = tensorflow.keras.layers.Dense(units = 1, activation = None, use_bias = None, name = "lin"+el[0]+outel)(model.inputs[el[0]])
                model.output_relation[outel].append(el[0])
                
            else: 
                if model.relations.get(el) is None:
                    model.relations[el+outel] = tensorflow.keras.layers.Dense(units = 1, activation = None, use_bias = None, name = "lin"+el+outel)(model.inputs[el])
                model.output_relation[outel].append(el)