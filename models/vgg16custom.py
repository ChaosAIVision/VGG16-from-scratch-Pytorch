import torch
import torch.nn as nn
from typing import Any, cast, Dict, List, Optional, Union
from pprint import pprint
    

# class CustomVGG16(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomVGG16,self).__init__()
#         self.layer1 = self.make_block(3, 64 , 3, 1, 1, 64 , False)
#         self.layer2 = self.make_block(64, 64 , 3, 1, 1, 64 , True)
#         self.layer3 = self.make_block(64, 128 , 3, 1, 1, 128 , False)
#         self.layer4 = self.make_block(128, 128 , 3, 1, 1, 128 , True)
#         self.layer5 = self.make_block(128, 256 , 3, 1, 1, 256 , False)
#         self.layer6 = self.make_block(256, 256 , 3, 1, 1, 256 , False)
#         self.layer7 = self.make_block(256, 256 , 3, 1, 1, 256 , True)
#         self.layer8 = self.make_block(256, 512 , 3, 1, 1, 512 , False)
#         self.layer9 = self.make_block(512, 512 , 3, 1, 1, 512 , False)
#         self.layer10 = self.make_block(512, 512 , 3, 1, 1, 512 , True)
#         self.layer11 = self.make_block(512, 512 , 3, 1, 1, 512 , False)
#         self.layer12 = self.make_block(512, 512 , 3, 1, 1, 512 , False)
#         self.layer13 = self.make_block(512, 512 , 3, 1, 1, 512 , True)
#         self.fc =nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(7*7*512,4096),
#             nn.ReLU()
#         )
#         self.fc1 =nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096,4096),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(4096, num_classes)
#         )


    
#     def make_block(self,in_chanels, out_chanels, kernel_size, stride, padding, num_BatchNorm, is_maxpooling):        
#         layer0 = nn.Sequential(
#             nn.Conv2d(in_chanels, out_chanels, kernel_size, stride, padding),
#             nn.BatchNorm2d(num_BatchNorm),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2)

#         )

#         layer1 = nn.Sequential(
#             nn.Conv2d(in_chanels, out_chanels, kernel_size, stride, padding),
#             nn.BatchNorm2d(num_BatchNorm),
#             nn.ReLU()
#              )

#         if is_maxpooling:
#             return layer0
#         else:
#             return layer1
        
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.layer7(x)
#         x = self.layer8(x)
#         x = self.layer9(x)
#         x = self.layer10(x)
#         x = self.layer11(x)
#         x = self.layer12(x)
#         x = self.layer13(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc(x)
#         x = self.fc1(x)
#         x = self.fc2(x)

#         return x

# model = CustomVGG16(10)
# print(model)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, )-> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers+= [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v= cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size= 3, padding= 1)
            if batch_norm:
                layers +=[conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace= True)]
            else:
                layers+= [conv2d,nn.ReLU(inplace= True)]
            in_channels = v
    
    return nn.Sequential(*layers)
    

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class CustomVGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes:int, init_weight: bool= True, dropout:float = 0.5) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveMaxPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(4096,1000),
            nn.Linear(1000,512),
            nn.Linear(512,num_classes)
        )
        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity= 'relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        
        return x
    

def vgg(cfg: str, batch_norm: bool,num_classes, **kwargs: Any) -> CustomVGG:
   
    model = CustomVGG(make_layers(cfgs[cfg], batch_norm=batch_norm),num_classes= num_classes, **kwargs)
    return model


if __name__ == "__main__":
    # Khởi tạo model với cấu hình D và num_classes = 10
    model = vgg('D', batch_norm=False, num_classes=10)
    pprint(model)
