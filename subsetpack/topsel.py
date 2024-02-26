#This class returns the number of top scored elements (as many defined) based on the importance weights/values
import numpy as np
import torch

class TopK(object):

	def __init__(self,scores,trainset,confdata):
		self.scores = scores
		self.selcount = confdata['num_datapoints']
		self.trainset = trainset
		self.trainb = confdata['trainbatch']

	def subsetpoints(self):

		trainloader = None

		if self.scores is None:
			print("No scores computed")
			subset_trvals = None
			subset_trindices = None
		else:
			sort_vals = -np.sort(-self.scores)
			n_sort_idx = np.argsort(-self.scores)
			subset_trindices = n_sort_idx[:self.selcount]
			subset_trvals = sort_vals[:self.selcount]
			
			subset_train = torch.utils.data.Subset(self.trainset, subset_trindices)
			trainloader = torch.utils.data.DataLoader(subset_train, batch_size=self.trainb, shuffle=True, num_workers=2)

		return subset_trvals,subset_trindices,trainloader