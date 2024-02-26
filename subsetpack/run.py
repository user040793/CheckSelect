import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.linalg import lstsq
from sklearn.preprocessing import normalize
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import random
import os
import argparse
import time
import copy
from numpy import linalg as LA
from subsetpack.helper import HelperFunc

class TrajSel(object):

    def __init__(self,train,test,model,helpobj,confdata):

        self.train_loader=train
        self.test_loader=test
        self.model = model
        self.subset_train = confdata['subtrain']
        self.rootpath = confdata['root_dir']
        self.resume = confdata['resume']
        self.epochs = confdata['epochs']
        self.numtp = confdata['num_trajpoint']
        self.unif_epoch_interval = self.epochs//self.numtp
        self.numfp = confdata['num_freqcp']
        self.csel_epoch_interval = confdata['num_freqep']
        self.csel_batch_interval = confdata['num_freqbatch']
        self.batchl = [i for i in range(0,len(self.train_loader),self.csel_batch_interval)]
        self.step_in_epoch = 0
        self.csel = confdata['csel']
        self.helpobj = helpobj
        self.count = 0
        self.best_acc = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
        if self.resume:
            checkpointpath = torch.load(self.rootpath+'checkpoint/ckpt.pth')
            self.model.load_state_dict(checkpointpath['model'])
            self.start_epoch = checkpointpath['epoch']+1
        else:
            self.start_epoch = 0
        self.lr = 0.1
        self.optimizer = optim.SGD(self.model.parameters(), self.lr)
        self.net1 = self.model
        self.model.to(self.device)
        self.net1.to(self.device)
        if not os.path.exists(self.rootpath):
            os.mkdir(self.rootpath)
        if not os.path.exists(self.rootpath+'checkpoint/'):
            os.mkdir(self.rootpath+'checkpoint/')
        if not(self.resume) and os.path.exists(self.rootpath+'misc/lastretain.pth'):
            os.remove(self.rootpath+'misc/lastretain.pth')



    def initialize(self):

        netv = {}
        for ix in self.batchl:
            netv[ix] = {}

        lossval = [0 for i in range(len(self.test_loader))]
        cz = [[] for i in range(len(self.test_loader))]
        czs = [[] for i in range(len(self.test_loader))]

        return netv,lossval,cz,czs

    def savemodel(self,netv=None,batchid=None,epoch=None,unif=False):

        if netv!=None:
            netv[batchid] = self.model.state_dict()
            self.count = self.count + 1

        if self.step_in_epoch!=0 and self.step_in_epoch%self.numfp==0 or self.step_in_epoch==len(self.train_loader)-1:
            state = {'model': self.model.state_dict(),'epoch': epoch}
            torch.save(state, self.rootpath+'checkpoint/ckpt.pth')
        if unif==True:
            torch.save(self.model.state_dict(), self.rootpath+'checkpoint/epoch_'+str(epoch)+'.pth')


    def test(self,epoch):

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, idx) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint_sub'):
                os.mkdir('checkpoint_sub')
            torch.save(state, './checkpoint_sub/ckpt.pth')
            self.best_acc = acc

        print(acc)
        print("Best Accuracy")
        print(self.best_acc)


    ########## Trains the model on a dataset; runs CheckSel at an interval epoch; stores the miscellaneous results and also the trajectory indices with their weights ###########
    def fit(self):

        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):

            self.count = 0
            trloss = 0

            netv, lossval, cz, czs = self.initialize()
            self.model.train()
            for batch_idx, (inputs, targets, idx) in enumerate(self.train_loader):

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.csel is True and batch_idx%self.csel_batch_interval==0 and epoch%self.csel_epoch_interval==0:
                    # Value function callback returning value function delta_{ind} and aggregate of estimate over all datapoints in the batch C_{ind}
                    lossval,cz,czs = self.helpobj.valuefunc_cb(epoch,batch_idx,inputs,targets,lossval,cz,czs,self.model,self.count)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if self.csel is True and batch_idx%self.csel_batch_interval==0 and epoch%self.csel_epoch_interval==0:
                    self.savemodel(netv=netv,batchid=batch_idx,epoch=epoch) #saving required model parameters

                self.savemodel(epoch=epoch)
                self.step_in_epoch+=1

                trloss = trloss + loss.item()
            # Saving uniformly spaced trajectories
            if self.subset_train is False and epoch%self.unif_epoch_interval==0:
                self.savemodel(epoch=epoch,unif=True)
            print('Epoch '+str(epoch)+', Loss '+str(trloss/len(self.train_loader)))
            #checksel callback function to run Algorithm 1: CheckSel
            self.helpobj.checksel_cb(lossval,czs,epoch,self.model,netv)

            if self.subset_train is True:
                self.test(epoch)