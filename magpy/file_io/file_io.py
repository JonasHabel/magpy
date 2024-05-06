import numpy as np
import os
import pickle
from ..lattice import ReciprocalLattice


class FileNamingConvention:
    def create_file_name(quantity_name, parameters: dict):
        raise NotImplementedError()
    
class DefaultFileNamingConvention(FileNamingConvention):
    def __init__(self, entry_separator, key_val_separator, quotation_char):
        self.entry_separator = entry_separator
        self.key_val_separator = key_val_separator
        self.quotation_char = quotation_char
        self.ndarray_formatter = str

    def quantity_to_string(self, quantity):
        if type(quantity) == np.ndarray:
            return "[" + ",".join(map(self.ndarray_formatter, quantity)) + "]"
        elif type(quantity) == ReciprocalLattice.MomentumPath:
            return "{" + ",".join(quantity.high_sym_point_labels) + ";" \
                + str(len(quantity.ks)) + "}"
        elif type(quantity) == tuple:
            return "(" + ",".join(map(self.quantity_to_string, quantity)) + ")"
        elif type(quantity) == list:
            return "[" + ",".join(map(self.quantity_to_string, quantity)) + "]"
        elif type(quantity) == str:
            q = self.quotation_char
            return f"{q}{quantity.split(os.sep)[-1]}{q}"
        else:
            return str(quantity)
        
    def key_to_string(self, key):
        return key.replace(self.key_val_separator, "-") \
                  .replace(self.entry_separator, "-")
    
    def create_key_val_pair_string(self, k, v):
        return self.key_to_string(k) + self.key_val_separator \
                + self.quantity_to_string(v)

    def create_file_name(self, quantity_name, parameters: dict):
        if not parameters:
            return quantity_name
        
        return quantity_name + self.entry_separator + self.entry_separator.join([
            self.create_key_val_pair_string(k, v) \
            for k, v in parameters.items()
        ])
    
    def create_glob(self, quantity_name, parameters: dict):
        if not parameters:
            return quantity_name
        
        return quantity_name + self.entry_separator + self.entry_separator.join([
            k if k in ["*"] else self.create_key_val_pair_string(k, v) \
            for k, v in parameters.items()
        ])


class FileIO:
    def __init__(self, folder: str,
            file_naming_convention=DefaultFileNamingConvention("_", "=", "`")):
        self.folder = folder
        self.file_naming_convention = file_naming_convention

    def __compose_file_path(self, quantity_name, parameters):
        return os.path.join(
            self.folder,
            self.file_naming_convention.create_file_name(
                quantity_name, parameters))
    
    def __get_file_name_params(meta_data, param_fields,
                               excluded_param_fields=()):
        param_fields = param_fields if param_fields is not None \
            else meta_data.keys()
        return dict(
            (k, v) for k, v in meta_data.items() \
            if k in param_fields and k not in excluded_param_fields \
            and v is not None)
   

    def save_data(self, data, quantity_name: str, meta_data: dict,
                  param_fields=None, excluded_param_fields=[]):
        params = FileIO.__get_file_name_params(
            meta_data, param_fields, excluded_param_fields)
        file_name = self.__compose_file_path(quantity_name, params)
        return FileIO.save_data_to_file(file_name, data, meta_data,
                                        list(params.keys()))
   

    def save_data_to_file(file_name: str, data: str, meta_data: dict,
                          param_keys: dict):
        with open(file_name, "wb") as f:
            pickle.dump(meta_data, f)
            pickle.dump(param_keys, f)
            pickle.dump(data, f)
        
        return file_name
    
    
    def load_data(self, quantity_name: str, meta_data: dict,
                  param_fields=None, excluded_param_fields=()):
        parameters = FileIO.__get_file_name_params(
            meta_data, param_fields, excluded_param_fields)
        file_name = self.__compose_file_path(quantity_name, parameters)
        return FileIO.load_data_from_file(file_name)
    
    def load_data_from_file(file_name: str):
        with open(file_name, "rb") as f:
            meta_data = pickle.load(f)
            parameter_fields = pickle.load(f)
            data = pickle.load(f)
        
        return (data, meta_data, parameter_fields, )
