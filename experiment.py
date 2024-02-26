from subsetpack.dataset import Dataset
from subsetpack.model import Model
from subsetpack.run import TrajSel
from subsetpack.helper import HelperFunc
from subsetpack.scoreval_checksel import DataValueCheckSel
from subsetpack.topsel import TopK
from subsetpack.diverselect import SimSel
import config_create
import os
import yaml
import json

def main():

	############ Run the config file to create a dictionary ##########
	#os.system('python config_create.py')
	
	'''with open("config.json", "r") as fp:
		confdata = json.load(fp) #holds the various configurable parameters needed ahead'''
		
	with open("config.yaml", "r") as fp:
		confdata = yaml.load(fp,Loader=yaml.FullLoader)

	########### Defining dataset class for loading the required data #############
	dataobj = Dataset(confdata)
	trainloader, testloader, trainset,testloader_s = dataobj.load_data()

	########### Defining model class for loading the model architecture ###########
	modelobj = Model()
	model = modelobj.ResNet18()

	########### Trains the model on the dataset and saves uniformly spaced trajectory points(model parameters)
	########### Or selected trajectories along with their importance weights using CheckSel
	helpobj = HelperFunc(trainloader,testloader,model,confdata)
	trajobj = TrajSel(trainloader,testloader,model,helpobj,confdata)
	trajobj.fit()

	########## Computes value for all training datapoints using selected trajectories from CheckSel #########

	dvalueobj = DataValueCheckSel(trainset,testloader,model,helpobj,confdata)
	scores,contrib = dvalueobj.scorevalue()

	########## Subset of datapoints obtained from the data values using SimSel or TopK procedure ##########
	if confdata['findsubset']:
		if confdata['simsel']:
			subsetobj = SimSel(trainset,scores,contrib,confdata)
			ind, subloader = subsetobj.subsetpoints_simsel()
		else:
			subsetobj = TopK(scores,trainset,confdata)
			val, ind, subloader  = subsetobj.subsetpoints()

		confdata['subtrain'] = True #subset training
		confdata['csel'] = False #No more CheckSel
		confdata['epochs']=300 #Set or keep the same as before
		helpobj = HelperFunc(trainloader,testloader,model,confdata)
		model = modelobj.ResNet18()
		trajobj = TrajSel(subloader,testloader,model,helpobj,confdata)
		trajobj.fit()

if __name__=='__main__':
	main()