import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import numpy as np 
import os
import argparse

from model.resnet import resnet101
from dataset.DogCat import DogCat

parser=argparse.ArgumentParser()
parser.add_argument('--num_workers',type=int,default=2)
parser.add_argument('--batchSize',type=int,default=64)
parser.add_argument('--nepoch',type=int,default=21)
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--gpu',type=str,default='7')
opt=parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

transform_train=transforms.Compose([
	transforms.Resize((256,256)),
	transforms.RandomCrop((224,224)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

transform_val=transforms.Compose([ 
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

trainset=DogCat('./data/train',transform=transform_train)
valset  =DogCat('./data/train',transform=transform_val)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)

model=resnet101(pretrained=True)
model.fc=nn.Linear(2048,2)
model.cuda()
optimizer=torch.optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=5e-4)
scheduler=StepLR(optimizer,step_size=3)
criterion=nn.CrossEntropyLoss()
criterion.cuda()

def train(epoch):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	model.train()
	for batch_idx,(img,label) in enumerate(trainloader):
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		out=model(image)
		loss=criterion(out,label)
		loss.backward()
		optimizer.step()
		print("Epoch:%d [%d|%d] loss:%f" %(epoch,batch_idx,len(trainloader),loss.mean()))

def val(epoch):
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total=0
	correct=0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			out=model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	print("Acc: %f "% ((1.0*correct.numpy())/total))

for epoch in range(opt.nepoch):
	train(epoch)
	val(epoch)
torch.save(model.state_dict(),'ckp/model.pth')
