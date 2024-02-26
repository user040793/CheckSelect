########## This class will return a subset of datapoints using Algorithm 3: SimSel after utilizing the contribution vectors obtained while computing values of datapoints ##########

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import time
import pickle
import sklearn
from sklearn.metrics import pairwise
from apricot import FacilityLocationSelection

class SimSel(object):


	def __init__(self,train,scores,contrib,confdata):

		self.trainset = train
		self.scores = scores
		self.dict_contrib = contrib
		self.selcount = confdata['num_datapoints']
		self.num_trajp = confdata['num_trajpoint']
		self.checksel = confdata['csel']
		self.resume = confdata['resume']
		self.trainb = confdata['trainbatch']
		if self.resume:
			checkpointpath = torch.load(self.rootpath+'checkpoint/ckpt.pth')
			self.model.load_state_dict(checkpointpath['model'])
			self.epochs = checkpointpath['epoch']+1
		else:
			self.epochs = confdata['epochs']


		if os.path.exists('new_dict.pkl'):
			os.remove('new_dict.pkl')

	############ Cosine similarity using contribution vectors from TracIn between each pair of datapoints sent in batches #########
	############ Higher similarity indicates similar contribution and hence redundancy #######
	def get_D(self,idx_list, S_dict, dict_contrib,epb):
	    # idx in idx_list are actual training set indices
		n = len(idx_list)
		D = np.zeros((n, n))
		sums = torch.zeros((n,n))
		contribckpt = []
		new_dict = {}
		if os.path.exists('new_dict.pkl')==False:
			for ckpt in epb:
				contrib_array = []
				for i in range(len(self.trainset)):
					contrib_array.append(dict_contrib[i][ckpt].cpu().detach().numpy())
				new_dict[ckpt] = np.array(contrib_array)

			with open('new_dict.pkl', 'wb') as fp:
				pickle.dump(new_dict, fp)

		with open('new_dict.pkl','rb') as fp:
			new_dict = pickle.load(fp)

		for ckpt in epb:
			contribs = new_dict[ckpt][np.array(idx_list)]
			contribs = np.squeeze(contribs, axis=2)
			contribckpt.append(pairwise.cosine_similarity(contribs)) #1100 x 1100

		for i in range(len(contribckpt)):

			sums = torch.add(sums,torch.from_numpy(contribckpt[i])) #1100 x 1100
		sums = torch.div(sums,len(contribckpt))
		sums = sums.numpy().copy()
		D = 1.0/(1.0+np.exp(-(sums)))
		return D

	############ Cosine similarity using contribution vectors from CheckSel between each pair of datapoints sent in batches #########
	############ Higher similarity indicates similar contribution and hence redundancy #######
	def get_D_cs(self,idx_list, S_dict, dict_contrib):
		# idx in idx_list are actual training set indices
		#uses contrib vectors to compute similarity 'D' in Simsel algorithm
		mapping_dict = {}
	    
		n = len(idx_list)
		total = 0
	    
		D = np.zeros((n, n))

		sums = torch.zeros((n,n))

		contribckpt = []
		new_dict = {}
		contrib_array = []

		if os.path.exists('new_dict.pkl')==False:
		
			for i in range(len(self.trainset)):
				contrib_array.append(dict_contrib[i].cpu().detach().numpy())
		
			new_dict = np.array(contrib_array)

			with open('new_dict.pkl', 'wb') as fp:
				pickle.dump(new_dict, fp)


		with open('new_dict.pkl','rb') as fp:
			new_dict = pickle.load(fp)

		contribs = new_dict[np.array(idx_list)]
		contribs = np.squeeze(contribs, axis=2)
		contribckpt = pairwise.cosine_similarity(contribs)
		
		D = 1.0/(1.0+np.exp(-(contribckpt)))

		return D

	######## Executes Algorithm 3 using similarity values D obtained from contribution vectors ############

	def subsetpoints_simsel(self):

		if self.scores is None or self.dict_contrib is None:
			print("No scores computed")
			Z_new = None
			trainloader = None
		else:
			if os.path.exists('new_dict.pkl'):
				os.remove('new_dict.pkl')

			epb = []
			for val in range(0,self.epochs,self.epochs//self.num_trajp):
				epb.append(val)

			batch_sz = min(self.selcount//10,500)
			S_dict = {}
			helpful_idx = np.argsort(-self.scores)
			pos_index = []

			for h in range(len(helpful_idx)):
				pos_index.append(helpful_idx[h])

			Z_0 = pos_index[:self.selcount]
			rest_idx = pos_index[self.selcount:]

			for i in range(0,len(rest_idx),batch_sz):
				print("Current Batch [",i,"-",i+batch_sz,"]")
				with open("result_"+str(self.selcount)+"_csel"+str(self.checksel)+".txt", 'a') as f:
					f.write("Current Batch [ " + str(i) + " - " + str(i+batch_sz) + " ]\n")

				B_i = rest_idx[i:i+batch_sz]
				Z_u_B = np.hstack((Z_0,B_i))
				facility_mapper = {}

				for c,zi in enumerate(Z_u_B):
					facility_mapper[c] = zi

				if self.checksel: #Computing D to be used in Simsel for CheckSel trajectories
					D = self.get_D_cs(Z_u_B, S_dict, self.dict_contrib)
				else: #Computing D to be used in Simsel for uniformly spaced trajectories
					D = self.get_D(Z_u_B, S_dict, self.dict_contrib,epb)

				modelfl = FacilityLocationSelection(self.selcount, 'precomputed', verbose=True)
				sobj = modelfl.fit(D)
				listinds = sobj.ranking
				Z_new = [facility_mapper[c] for c in listinds[:self.selcount]]
				Z_0 = Z_new
				with open("result_"+str(self.selcount)+"_csel"+str(self.checksel)+".txt", 'a') as f:
					f.write(str(list(Z_new))+"\n")

			subset_train = torch.utils.data.Subset(self.trainset, list(Z_new))
			trainloader = torch.utils.data.DataLoader(subset_train, batch_size=self.trainb, shuffle=True, num_workers=2)

		return Z_new, trainloader
