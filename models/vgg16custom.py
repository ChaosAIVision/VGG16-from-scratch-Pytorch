import torch
import torch.nn as nn


    

class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16,self).__init__()
        self.layer1 = self.make_block(3, 64 , 3, 1, 1, 64 , False)
        self.layer2 = self.make_block(64, 64 , 3, 1, 1, 64 , True)
        self.layer3 = self.make_block(64, 128 , 3, 1, 1, 128 , False)
        self.layer4 = self.make_block(128, 128 , 3, 1, 1, 128 , True)
        self.layer5 = self.make_block(128, 256 , 3, 1, 1, 256 , False)
        self.layer6 = self.make_block(256, 256 , 3, 1, 1, 256 , False)
        self.layer7 = self.make_block(256, 256 , 3, 1, 1, 256 , True)
        self.layer8 = self.make_block(256, 512 , 3, 1, 1, 512 , False)
        self.layer9 = self.make_block(512, 512 , 3, 1, 1, 512 , False)
        self.layer10 = self.make_block(512, 512 , 3, 1, 1, 512 , True)
        self.layer11 = self.make_block(512, 512 , 3, 1, 1, 512 , False)
        self.layer12 = self.make_block(512, 512 , 3, 1, 1, 512 , False)
        self.layer13 = self.make_block(512, 512 , 3, 1, 1, 512 , True)
        self.fc =nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512,4096),
            nn.ReLU()
        )
        self.fc1 =nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )


    
    def make_block(self,in_chanels, out_chanels, kernel_size, stride, padding, num_BatchNorm, is_maxpooling):        
        layer0 = nn.Sequential(
            nn.Conv2d(in_chanels, out_chanels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_BatchNorm),
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )

        layer1 = nn.Sequential(
            nn.Conv2d(in_chanels, out_chanels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_BatchNorm),
            nn.ReLU()
             )

        if is_maxpooling:
            return layer0
        else:
            return layer1
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

model = CustomVGG16(10)
print(model)


        