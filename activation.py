from tqdm import tqdm
import numpy as np

import torch

import tool as tool


class Acti_base:
    def __init__(self, model):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.init_variable()

    def init_variable(self):
        raise NotImplementedError
        
    def calculate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')

    def assess(self, data_loader):
        for data in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                try:
                    data = data.to(self.device)
                except:
                    pass
            self.step(data)

    def step(self, data):
        acti_dict, model_out = self.calculate(data)
        self.acti_dict = acti_dict
        return model_out

    def update(self, all_cove_dict, delta=None):
        self.acti_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current



class Acti(Acti_base):
    def init_variable(self):
        self.acti_dict = {}
        self.current = 0
        
    def count_level(self, scaled_output):
        return scaled_output

    def calculate(self, data):
        acti_dict = {}
        layer_output_dict, model_out = tool.get_layer_output(self.model, data)     
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = layer_output
            acti_dict[layer_name] = self.count_level(scaled_output)
        return acti_dict, model_out

    def save(self, path):
        torch.save(self.acti_dict, path)

    def load(self, path):
        self.acti_dict = torch.load(path)
    
    def get_acti_dict(self):
        return self.acti_dict
    
    def get_abs_output(self):
        layer_output_dict = {}
        for layer_name in self.acti_dict.keys():
            layer_output_dict[layer_name] = self.acti_dict[layer_name]
        return layer_output_dict

    def clear_activations(self):
        self.acti_dict = {}
        
