######### This class will compute values/importance weights for all data points using selected trajectories from CheckSel #############

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np
import copy
import os
import argparse
import time
import pickle


class DataValueCheckSel(object):

    def __init__(self,train,test,model,helperobj,confdata):

        self.trainset = train
        self.testloaderb = test
        self.model = model
        self.rootpath = confdata['root_dir']
        self.featdim = confdata['featuredim']
        self.bsize = confdata['trainbatch']
        self.keeprem = confdata['retain_scores']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resume = confdata['resume']
        self.fdim = confdata['featuredim']
        self.layer = self.model._modules.get('avgpool')
        checkpointpath = torch.load(self.rootpath+'checkpoint/ckpt.pth')
        self.resf = self.model
        self.resf.load_state_dict(checkpointpath['model'])
        self.resf.to(self.device)
        self.resf.eval()
        self.helperobj = helperobj

    ############# Finds the neighbour using computed features #############
    def findnearest(self,node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
        return np.argmin(dist_2)

    ############ Computes values for datapoints assigned to the selected trajectories during training ###########
    ############ For e.g. trajectory epoch10_batch15 will be associated with batch15 data where the batch id is already fixed during initial dataloading #########

    '''def get_feature_single(self,image):
        embed = torch.zeros(self.fdim)
        def copy_data(m,i,o):
            embed.copy_(o.data.reshape(o.data.size(1)))
        h = self.layer.register_forward_hook(copy_data)
        self.resf(image)
        h.remove()
        return embed'''

    def get_feature(self,image):
        embed = torch.zeros((image.shape[0],self.fdim))
        def copy_data(m,i,o):
            embed.copy_(o.data.reshape(o.data.size(0),o.data.size(1)))
        h = self.layer.register_forward_hook(copy_data)
        self.resf(image)
        h.remove()
        return embed

    def score_trajbatch(self):

        cpind = []
        alphaval = []
        indcp = np.load(self.rootpath+'trajp_value_indices.npy')
        for ind in range(indcp.shape[0]):
            cpind.append((indcp[ind][1],indcp[ind][2]))
            alphaval.append(indcp[ind][0])

        subset_trindices = []
        btlist = set()
        cps = os.listdir(self.rootpath+'trajp/')
        res = [ 0 for i in range(len(self.trainset)) ]
        cpv = [ [] for i in range(len(self.trainset)) ]
        repeat = [ 0 for i in range(len(self.trainset)) ]
        fea = {}
        testgr = {}
        dict_contrib = {}
        ind = 0
        contribv = {}



        for i in range(len(self.trainset)):
            contribv[i] = None

        for i in range(len(self.trainset)):
            dict_contrib[i] = None
            
        numind = []
        modelparam = {}
        for ckpt in cps:
                ep = int(ckpt.split('_')[1].split('_')[0])
                bt = int(ckpt.split('_')[3].split('.')[0])
                start = ((self.bsize-1)*bt)+bt
                end = start + (self.bsize-1)
                for s in range(start,end+1):
                    numind.append(s)

        fv = np.zeros((len(numind),self.featdim)) #Number of instances through selected checkpoints

        #Computes scores for instances in B (Eq. 8)
        for ckpt in cps:
                # considering your ckpt name is 'epoch_<epoch value>_batch_<batch value>.pth'.
                net = torch.load(self.rootpath+'trajp/'+ckpt)
                modelparam[ckpt]=copy.deepcopy(net)
                subset_pickindices =[]
                ep = int(ckpt.split('_')[1].split('_')[0])
                bt = int(ckpt.split('_')[3].split('.')[0])
                alpha = alphaval[cpind.index((ep,bt))]
                start = ((self.bsize-1)*bt)+bt
                end = start + (self.bsize-1)
                test_gradb = None
                for testi, data in enumerate(self.testloaderb,0):
                    inputs, targets, idx = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    if test_gradb is None :
                        test_gradb = self.helperobj.get_grad(inputs,targets, net)
                        test_gradb = test_gradb.unsqueeze(0)
                    else:
                        test_gradb = torch.cat((test_gradb, self.helperobj.get_grad(inputs,targets, net).unsqueeze(0)), axis = 0)

                testgr[ckpt] = test_gradb

                for s in range(start,end+1):
                    subset_trindices.append(s)
                    subset_pickindices.append(s)
                    cpv[s].append((ep,bt))


                subsetcp = torch.utils.data.Subset(self.trainset, subset_pickindices)
                trainsubloader = torch.utils.data.DataLoader(subsetcp, batch_size=1, shuffle=False, num_workers=2)

                
                for batch_idx, (inputs, targets, idx) in enumerate(trainsubloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    fv[ind] = self.get_feature(inputs).squeeze(0)
                    ind = ind + 1
                    train_grad_single = self.helperobj.get_grad(inputs, targets, net).unsqueeze(0)
                    tempf = test_gradb*train_grad_single
                    temp = (tempf + 0.5*(tempf*tempf))/len(trainsubloader)  #Element wise multiplication
                    repeat[subset_pickindices[batch_idx]]+=1
                    res[subset_pickindices[batch_idx]] += (alpha*(torch.sum(temp).item()))/repeat[subset_pickindices[batch_idx]] #score
                    if contribv[subset_pickindices[batch_idx]] is None:
                        contribv[subset_pickindices[batch_idx]] = (alpha*(torch.sum(temp,(1,2)).unsqueeze(1)))#contribution vector
                    else:
                        contribv[subset_pickindices[batch_idx]] += (alpha*(torch.sum(temp,(1,2)).unsqueeze(1)))

                    if dict_contrib[subset_pickindices[batch_idx]] is None:
                        dict_contrib[subset_pickindices[batch_idx]] = {}
                        dict_contrib[subset_pickindices[batch_idx]] = contribv[subset_pickindices[batch_idx]]/repeat[subset_pickindices[batch_idx]]
                    else:
                        
                        dict_contrib[subset_pickindices[batch_idx]] = contribv[subset_pickindices[batch_idx]]/repeat[subset_pickindices[batch_idx]]
        
        return cpind,alphaval,fv,subset_trindices,cpv,testgr,res,dict_contrib,modelparam,len(trainsubloader)


    ############ Computes values for datapoints not assigned to the selected trajectories(or out of batch)during training ###########

    def score_oobatch(self,cpind,alphaval,modelparam,fv,subset_trindices,cpv,testgr,res,dict_contrib,lensub):

        rem_ind = set(np.arange(len(self.trainset))).difference(set(subset_trindices))
        rem_ind = list(sorted(list(rem_ind)))
        subset_rem = torch.utils.data.Subset(self.trainset, rem_ind)
        trainrem = torch.utils.data.DataLoader(subset_rem, batch_size=100, shuffle=True, num_workers=2)
        actind = 0
   
        #Computes scores for instance e in Z \ B by finding neighbour n from B
        for batch_idx, (inp, targ, idx) in enumerate(trainrem):
           ninds = []
           inp, targ = inp.to(self.device), targ.to(self.device)
           featime = time.time()
           contrib_vec = None
           feari = self.get_feature(inp)
           ninds = np.argmin(manhattan_distances(feari, fv),axis=1)
           for index in range(len(ninds)):
                   acind = subset_trindices[ninds[index]]
                   epbt = cpv[acind]
                   contrib_vec = None
                   for el in range(len(epbt)):
                       ep,bt = epbt[el] #finding checkpoint to be used for computing score as per Eq. 8
                       alpha = alphaval[cpind.index((ep,bt))]
                       net = modelparam['epoch_'+str(ep)+'_batch_'+str(bt)+'.pth']
                       test_gradb = testgr['epoch_'+str(ep)+'_batch_'+str(bt)+'.pth']
                       train_grad_single = self.helperobj.get_grad(inp[index].unsqueeze(0), targ[index].unsqueeze(0), net).unsqueeze(0)
                       tempf = test_gradb*train_grad_single
                       temp = (tempf + 0.5*(tempf*tempf))/lensub
                       res[rem_ind[actind]] += alpha*(torch.sum(temp).item()) # score

                       if contrib_vec is None:
                          contrib_vec = (alpha*(torch.sum(temp,(1,2)).unsqueeze(1)))# contrib vector
                       else:
                          contrib_vec += (alpha*(torch.sum(temp,(1,2)).unsqueeze(1)))# contrib vector


                       if dict_contrib[rem_ind[actind]] is None:
                          dict_contrib[rem_ind[actind]] = {}
                          dict_contrib[rem_ind[actind]] = contrib_vec
                       else:
                          dict_contrib[rem_ind[actind]] = contrib_vec
                   res[rem_ind[actind]] = res[rem_ind[actind]]/len(epbt)
                   dict_contrib[rem_ind[actind]] = dict_contrib[rem_ind[actind]]/len(epbt)
                   actind = actind + 1

        with open('./influence_scores.npy', 'wb') as f:
            np.save(f, np.array(res))
        with open('./dict_contrib.pkl', 'wb') as fp:
            pickle.dump(dict_contrib, fp)

        return np.array(res),dict_contrib


    ############ If computed data values (influence score, contribution vectors) are not to be used, the existing ones are removed and calculated again using selected trajectories ##############
    ############ Returns score and contribution vectors ##########
    def scorevalue(self):

        ########### If one has resumed, scores have to be recomputed. In order to avoid any eventuality, making retain_scores false explicitly############# 
        if self.resume:
            self.keeprem = False #If training is resumed from some point, data values/scores have to be recomputed
 
        if not self.keeprem:
            if os.path.exists('./influence_scores.npy'):
                os.remove('./influence_scores.npy')
            if os.path.exists('./dict_contrib.pkl'):
                os.remove('./dict_contrib.pkl')

            cpind,alphaval,fv,subset_trindices,cpv,testgr,res,dict_contrib,modelparam,lensub = self.score_trajbatch()            
            scores,contrib = self.score_oobatch(cpind,alphaval,modelparam,fv,subset_trindices,cpv,testgr,res,dict_contrib,lensub)

        else:
            if os.path.exists('influence_scores.npy'):
                scores = np.load('./influence_scores.npy')
            else:
                scores = None #retain_scores True even if no scores are kept recomputed
                
            if os.path.exists('dict_contrib.pkl'):
                with open('dict_contrib.pkl','rb') as fp:
                    contrib = pickle.load(fp)
            else:
                contrib = None

        return scores,contrib