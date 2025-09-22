import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import constants

def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

def is_valid(module):
    return (isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv3d)
            or isinstance(module, nn.RNN)
            or isinstance(module, nn.LSTM)
            or isinstance(module, nn.GRU)
            )

'''
path_list: 
'''
def iterate_module(name, module, name_list, module_list, path, path_list=[]):
    if is_valid(module):
        return name_list + [name], module_list + [module], path_list + [path]
    else:
        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list, path_list = \
                    iterate_module(child_name, child_module, name_list, module_list, path=path + '.' + child_name, path_list=path_list)
        return name_list, module_list, path_list

def get_model_layers(model):
    layer_dict = {}
    for name, module in model.named_children():
        path_list = []
        name_list, module_list, path_list = iterate_module(name, module, [], [], path=name, path_list=path_list)
        assert len(name_list) == len(module_list)
        assert len(name_list) == len(path_list)
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            layer_dict[path_list[i]] = module
    return layer_dict

def get_layer_output_sizes(model, data, pad_length=constants.PAD_LENGTH):   
    output_sizes = {}
    hooks = []
    name_counter = {}
    layer_dict = get_model_layers(model)
    def create_hook(name):
        def hook(module, input, output):
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
                output_sizes[name] = [output[0].size(2)]
            elif output.dim() == 3:
                output_sizes[name] = list([output.size()[2] * output.size()[1]])
            else:
                output_sizes[name] = list(output.size()[1:])
        return hook

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(create_hook(name)))    
    try:
        model(**data)
    except:
        model(data)
    finally:
        for h in hooks:
            h.remove()

    unrolled_output_sizes = {}
    for k in output_sizes.keys():
        if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
            for i in range(pad_length):
                unrolled_output_sizes['%s-%d' % (k, i)] = output_sizes[k]
        else:
            unrolled_output_sizes[k] = output_sizes[k]
    return unrolled_output_sizes

def get_layer_output(model, data, pad_length=constants.PAD_LENGTH):
    with torch.no_grad():
        name_counter = {}        
        layer_output_dict = {}
        layer_dict = get_model_layers(model)
        def create_hook(name):
            def hook(module, input, output):
                class_name = module.__class__.__name__
                if class_name not in name_counter.keys():
                    name_counter[class_name] = 1
                else:
                    name_counter[class_name] += 1
                
                if ('RNN' in class_name) or ('LSTM' in class_name) or ('GRU' in class_name):
                    layer_output_dict[name] = output[0]
                else:
                    layer_output_dict[name] = output
            return hook

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(create_hook(layer)))
        final_out = model(**data)
        for h in hooks:
            h.remove()

        unrolled_layer_output_dict = {}
        for k in layer_output_dict.keys():
            if ('RNN' in k) or ('LSTM' in k) or ('GRU' in k):
                assert pad_length == len(layer_output_dict[k])
                for i in range(pad_length):
                    unrolled_layer_output_dict['%s-%d' % (k, i)] = layer_output_dict[k][i]
            else:
                unrolled_layer_output_dict[k] = layer_output_dict[k]

        for layer, output in unrolled_layer_output_dict.items():
            if len(output.size()) == 4: 
                output = output.mean((2, 3))
            if len(output.size()) == 3: 
                output = output
            unrolled_layer_output_dict[layer] = output.detach()
        return unrolled_layer_output_dict, final_out