from . import define, model

def Model(args: str) -> list:
    model_info = []
    for layer in args.split(' '):
        layer_name = layer.split('=')[0]
        layer_value = layer.split('=')[1]
        if layer_name == "Flatten":
            if layer_value.strip('()') == "":
                model_info.append(define.Flatten(shape=()))
                continue
            shape = tuple(map(int, layer_value.strip('()').split('x')))
            model_info.append(define.Flatten(shape=shape))
        elif layer_name == "Dense":
            units, activation = [x.strip() for x in layer_value.strip('()').split(',')]
            units = int(units)
            activation = activation.strip("'")
            model_info.append(define.Dense(units=units, activation=activation))
        elif layer_name == "Dropout":
            rate = float(layer_value.strip('()'))
            model_info.append(define.Dropout(rate=rate))
        elif layer_name == "MaxPooling2D":
            shape = tuple(map(int, layer_value.strip('()').split('x')))
            model_info.append(define.MaxPooling2D(max=shape))
        elif layer_name == "Conv2D":
            shape = layer_value.strip('()').split(',')
            if len(shape) == 3 or shape[3] == "":
                model_info.append(define.Conv2D(units=int(shape[0]),size=tuple(map(int, shape[1].strip('()').split('x'))),activation=shape[2].strip("'"),shape=()))
            else:
                model_info.append(define.Conv2D(units=int(shape[0]),size=tuple(map(int, shape[1].strip('()').split('x'))),activation=shape[2].strip("'"),shape=tuple(map(int, shape[3].strip('()').split('x')))))
    return model_info

def Compile(args: str) -> model.CompileD:
    compile_info = model.CompileD(
        optimizer='',
        loss='',
        metrics=[]
    )
    for layer in args.split(' '):
        layer_name = layer.split('=')[0]
        layer_value = layer.split('=')[1]
        if layer_name == "Optimizer":
            compile_info.optimizer = layer_value.strip('()').strip("'")
        elif layer_name == "Loss":
            compile_info.loss = layer_value.strip('()').strip("'")
        elif layer_name == "Metrics":
            compile_info.metrics = layer_value.strip('()').strip("'").split(',')
    return compile_info