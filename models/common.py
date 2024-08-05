import os
import torch
import onnxruntime as ort

# Kiểm tra nếu có CUDA
cuda_available = torch.cuda.is_available()

if cuda_available:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.tools
    import tensorrt as trt

from vgg16custom import vgg
from utils.general import ManagerDataYaml


class ModelManagement():
    def __init__(self, name:str, version:str, weights_path:str, data_yaml: str,  batch_norm: bool = None):
        self.name = name
        self.version = version
        self.format = None
        self.weights_path = weights_path
        self.model = None
        self.batch_norm  = batch_norm
        data_yaml_manage = ManagerDataYaml(data_yaml)
        data_yaml_manage.load_yaml()
        self.num_classes = data_yaml_manage.get_properties(key='num_classes')

    def model_init_structure(self):
        try:
              # FOR MODEL VGG16
              if self.name == "vgg16":
                   self.model = vgg(self.version , self.num_classes, batch_norm= self.batch_norm)
                   return self.model
                   
        except:
             print('we are not support {self.name} ')
         

    def check_format(self):
            _, extension = os.path.splitext(self.weights_path)
            try:
                if extension == '.pt':
                    self.format = 'pytorch'
                elif extension == '.onnx':
                    self.format = 'onnx'
                elif extension == '.trt' or '.engine':
                    self.format =  'engine'
            except:
                 print(f'Cannot support {extension} format, we support only .pt .onnx and .trt .engine format')
    def load_pytorch_model(self):
        print('Your model must be save with three dictionaries: model_state_dict to save model parameters, epochs to save epochs and optimizer_state_dict to save parameters of optimizers')
        checkpoint = torch.load(self.weights_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return self.model
    
    def load_tensorrt_model(self):
        if not cuda_available:
            raise RuntimeError("CUDA is not available. Cannot load TensorRT model.")
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.weights_path, 'rb') as f:
            engine_data = f.read()

        self.model = runtime.deserialize_cuda_engine(engine_data)
        return self.model
    
    def load_onnx_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda_available else ['CPUExecutionProvider']
        self.model = ort.InferenceSession(self.weights_path, providers=providers)
        return self.model

    def loading_weight(self):
        self.check_format()
        if self.format == 'pytorch':
             self.model_init_structure()
             self.load_pytorch_model()
        elif self.format == 'onnx':
             self.load_onnx_model()
        elif self.format == 'engine':
             self.load_tensorrt_model()
        return self.model
             
        


         
        
    

if __name__ == ('__main__'):
     path = 'your_path/best.pt'
     model = ModelManagement('vgg16', 'D', path)
     model.check_format()
     




