
import torch
import torch.nn as nn

class VGG16(nn.Module):
    
    def __init__(self, num_classes= 5):
        super(VGG16,self).__init__()
        
        # Con layers
        self.features = nn.Sequential(
            # Block1
            nn.Conv2d(in_channels=3,out_channels= 64,kernel_size=3,padding=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=64,out_channels= 64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),            
            
            # Block2
            nn.Conv2d(in_channels=64,out_channels= 128,kernel_size=3,padding=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=128,out_channels= 128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),            
            
            # Block3
            nn.Conv2d(in_channels=128,out_channels= 256,kernel_size=3,padding=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=256,out_channels= 256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),               
            
            # Block 4
            nn.Conv2d(in_channels=256,out_channels= 512,kernel_size=3,padding=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=512,out_channels= 512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),               
            
            # Block 5
            nn.Conv2d(in_channels=512,out_channels= 512,kernel_size=3,padding=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=512,out_channels= 512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   
            
            )
        # 512 x 7 x 7                                               
        
        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self,x):
        x = self.features(x)
        x =x.view(x.size(0),-1) # Flaten the tensor
        x = self.classifier(x)
        return x

## using VGG16

model = VGG16()
print(model)

# Data
# Dataloader
# Transformer

# Loss fuction: Crossentropy
# Optimizer: Backprobagation adam

# For loop numer of epochs
# loop 2 reading the images in batch

from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torch import optim


# data
# train/
#     ├── dogs/
#     ├── cats/
#     └── birds/

# Transformer
imgsize = 224
train_dir = "D:/01-DATA/dum"

batch_size = 4
learing_rate = 0.001
epochs = 10

trans = transforms.Compose([
    transforms.Resize((imgsize,imgsize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225] )    
    ] )
# Dataset

train_data = datasets.ImageFolder(root=train_dir,transform=trans )
valid_data =  datasets.ImageFolder(root=train_dir,transform=trans )

train_loader = DataLoader(dataset= train_data, batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(dataset= valid_data, batch_size=batch_size,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=4).to(device=device)

# loss and optimizer


Critiria_loss_ = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learing_rate)

for epoch in range(epochs):
    model.train()
    runningLoss = 0.0
    iter = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        
        out = model(images)
        loss = Critiria_loss_(out,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
        iter +=1
        if iter %10 ==0:
            print(f'[{epoch+1}, {epochs}]  loss : {runningLoss}')
    


PATH = './model.pth'
torch.save(model.state_dict(), PATH)
        
# TEsting dat

correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')       
        
        
        
