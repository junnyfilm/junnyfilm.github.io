---
layout: post
title:  HRD Fault diagnosis using hybrid method 
date:   2022-11-04 16:06:59 +0900
tag: [Project]
---

# HRD Fault diagnosis using hybrid method of deep learning and signal processing

 ![hrd1](../img/hrd/hrd1.png)
 ![hrd2](../img/hrd/hrd2.png)
 ![hrd3](../img/hrd/hrd3.png)
 ![hrd4](../img/hrd/hrd4.png)
 ```
 class customdataset(Dataset):
    def __init__(self, data, label): 
        super().__init__()
        self.pin_data=data[:,0,:].unsqueeze(1)   
        self.po_data=data[:,1,:].unsqueeze(1)
        self.pdin_data=data[:,2,:].unsqueeze(1)
        self.label=label
        
    def __len__(self):
        return len(self.label)
  
    def __getitem__(self, idx):
        pin_data = self.pin_data[idx]
        po_data = self.po_data[idx]
        pdin_data = self.pdin_data[idx]
        label = self.label[idx] 
                
        return  pin_data.to(device).float(),po_data.to(device).float(),pdin_data.to(device).float(), label.to(device)
class customdataset_ts(Dataset):
    def __init__(self, data): 
        super().__init__()
        self.pin_data=data[:,0,:].unsqueeze(1)   
        self.po_data=data[:,1,:].unsqueeze(1)
        self.pdin_data=data[:,2,:].unsqueeze(1)
        
    def __len__(self):
        return len(self.pin_data)
  
    def __getitem__(self, idx):
        pin_data = self.pin_data[idx]
        po_data = self.po_data[idx]
        pdin_data = self.pdin_data[idx]
                
        return  pin_data.to(device).float(),po_data.to(device).float(),pdin_data.to(device).float()
def loaders(data,label, data2,label2):
    train_set, valid_set, train_label, valid_label = train_test_split(data, label, train_size=0.9, random_state=1)
    traindataset = customdataset(train_set, train_label)
    validdataset = customdataset(valid_set, valid_label)
    traindataloader1 = DataLoader(traindataset, batch_size=32, shuffle=True, drop_last=True )
    validdataloader1 = DataLoader(validdataset, batch_size=32, shuffle=True, drop_last=True )
    testdataset = customdataset(data2, label2)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False, drop_last=False )
    return traindataloader1,validdataloader1,testloader
 ```

 ![hrd5](../img/hrd/hrd5.png)
 '''
 class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool,self).__init__()
    def forward(self,x):
        return x.mean(axis=-1) 
    
class Feature_extractor1(nn.Module):

    def __init__(self):
        super(Feature_extractor1, self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=1, stride = 1, padding=0),
                                 nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=3, stride = 1, padding=1),
                                 nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=5, stride = 1, padding=2),
                                 nn.ReLU())

        
        
    def forward(self, x):
        x = self.bn(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        return x1,x2,x3
class Feature_extractor2(nn.Module):

    def __init__(self):
        super(Feature_extractor2, self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=1, stride = 1, padding=0),
                                 nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=3, stride = 1, padding=1),
                                 nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=5, stride = 1, padding=2),
                                 nn.ReLU())

        
        
    def forward(self, x):
        x = self.bn(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        return x1,x2,x3
class Feature_extractor3(nn.Module):

    def __init__(self):
        super(Feature_extractor3, self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=1, stride = 1, padding=0),
                                 nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=3, stride = 1, padding=1),
                                 nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(1, 3, kernel_size=5, stride = 1, padding=2),
                                 nn.ReLU())

        
        
    def forward(self, x):
        x = self.bn(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        return x1,x2,x3


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv,self).__init__()
        self.bn_concat = nn.BatchNorm1d(30)

        self.bottle_1 = nn.Conv1d(30,128,kernel_size=1, stride=1, bias=False)
        self.bottle_2 = nn.Conv1d(128,64, kernel_size=3, stride=1, bias=False, padding=1)

        self.bottle_3 = nn.Conv1d(30,128,kernel_size=3, stride=1, bias=False, padding=1)
        self.bottle_4 = nn.Conv1d(128, 64, kernel_size=1, stride=1, bias=False)

        self.bn_bottleneck24 = nn.BatchNorm1d(30+64+64)

        self.bottle_5 = nn.Conv1d(30+64+64,256,kernel_size=1, stride=1, bias=False)
        self.bottle_6 = nn.Conv1d(256,128,kernel_size=3, stride=1, bias=False, padding=1)

        self.bn_output = nn.BatchNorm1d(128)
        self.conv_out = nn.Conv1d(128, 64, kernel_size=1, stride=1, bias=False)
        self.relu =  nn.ReLU()
        self.gap = GlobalAvgPool()
        

        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.fc2 =nn.Linear(32, 11)

    def forward(self, inputs):
        concat = self.relu(inputs)
        bn_concat = self.bn_concat(concat)

        bottle_1 = self.bottle_1(bn_concat)
        bottle_1 = self.relu(bottle_1)
        bottle_2 = self.bottle_2(bottle_1)
        bottle_2 = self.relu(bottle_2)

        bottle_3 = self.bottle_3(bn_concat)
        bottle_3 = self.relu(bottle_3)
        bottle_4 = self.bottle_4(bottle_3)
        bottle_4 = self.relu(bottle_4)

        bottle24_concat = torch.cat([bn_concat, bottle_2, bottle_4],axis=1)
        bottle24_concat = self.bn_bottleneck24(bottle24_concat)

        bottle_5 = self.bottle_5(bottle24_concat)
        bottle_5 = self.relu(bottle_5)
        bottle_6 = self.bottle_6(bottle_5)
        bottle_6 = self.relu(bottle_6)

        conv_out = self.bn_output(bottle_6)
        out = self.conv_out(conv_out)
        out = self.relu(out)
        outview = self.gap(out)

        out = self.fc1(outview)
        out = self.fc2(out)

        return out,outview


feature_extractor1 = Feature_extractor1().to(device)
feature_extractor2 = Feature_extractor2().to(device)
feature_extractor3 = Feature_extractor3().to(device)
SimpleConvmodel = SimpleConv().to(device)

'''

 ![hrd6](../img/hrd/hrd6.png)
 ![hrd7](../img/hrd/hrd7.png)
 ![hrd8](../img/hrd/hrd8.png)
 ![hrd9](../img/hrd/hrd9.png)
 ![hrd10](../img/hrd/hrd10.png)
 ![hrd11](../img/hrd/hrd11.png)
 ![hrd12](../img/hrd/hrd12.png)
 ![hrd13](../img/hrd/hrd13.png)
 ![hrd14](../img/hrd/hrd14.png)
 ![hrd15](../img/hrd/hrd15.png)
 ![hrd16](../img/hrd/hrd16.png)
 ![hrd17](../img/hrd/hrd17.png)
 ![hrd18](../img/hrd/hrd18.png)
 ![hrd19](../img/hrd/hrd19.png)