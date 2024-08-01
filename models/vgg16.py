import torch 
from pprint import pprint

from torchvision.models.vgg import vgg16, VGG16_Weights

model = vgg16(VGG16_Weights)
pprint(model)