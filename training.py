from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from DataSetCreation import *
from TrainingModel import *
from ModelCreation import *

Image_size = 16
composed = transforms.Compose([transforms.Resize((Image_size,Image_size)), transforms.ToTensor()])
train_data = Dataset(train=True,transform=composed)
val_data = Dataset(train=False,transform=composed)

train_loader = DataLoader(train_data,batch_size=100)
val_loader = DataLoader(val_data,batch_size=100)
print(len(train_data))
print(len(val_data))
print(len(val_loader))

N_test = len(val_data)
model = CNN()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
criterion = nn.CrossEntropyLoss()
path = "entire_model.pt"
epochs = 15
modelobj = train_model(model=model,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,criterion=criterion,epochs=epochs,N_test=N_test,path=path)
accuracy , loss, cost = modelobj.train()

plt.plot(accuracy,'r',label="Accuracy")
plt.title("Accuracy")
plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig("Accuracy.png",dpi=100)

plt.plot(loss,'r',label="Loss")
plt.title("Loss")
plt.legend()
fig2 = plt.gcf()
plt.show()
plt.draw()
fig2.savefig("Loss.png",dpi=100)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.set_xlabel('epoch', color=color)
ax2.plot(accuracy, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()

fig3 = plt.gcf()
plt.show()
plt.draw()
fig3.savefig("Accuracy and Cost",dpi=100)

#for x,y in train_loader:
#    plt.imshow(x.numpy().reshape(Image_size,Image_size),cmap='gray')
#    plt.show()