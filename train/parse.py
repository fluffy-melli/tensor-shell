def Setting(args: str):
    info = {}
    for layer in args.split(' '):
        layer_name = layer.split('=')[0]
        layer_value = layer.split('=')[1]
        if layer_name == "Mode":
            info['Mode'] = layer_value.strip("'")
        elif layer_name == "Shape":
            info['Shape'] = tuple(map(int, layer_value.strip('()').split('x')))
        elif layer_name == "Batch":
            info['Batch'] = int(layer_value.strip('()'))
    return info