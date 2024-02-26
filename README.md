# CheckSelect

# Getting Started

Run the following commands to setup the environment:

```
git clone https://github.com/user040793/CheckSelect.git

cd Checkpoint-Select

conda create --name checksel --file requirements.txt
```

After the environment gets installed,

```
conda activate checksel
```
# Current Setup

<i>Dataset:</i> CIFAR-10

<i>Model:</i> ResNet-18

The default parameters are provided in ```config.yaml```. One can vary the parameters by running ```python config_create.py```

In order to obtain selected checkpoints followed by data valuation and training the subset of data, run

```python experiment.py```

# Code Structure

The different modules in which this code repository has been organised is:

1. Dataset (dataset.py) - One can set up their own dataset as needed.

2. Model (model.py) - Different models can be added here.

3. Trajectory selection during training (run.py) - Selects checkpoints/trajectories along with training.

4. Value function definition and CheckSel algorithm (helper.py) - Different other value functions can be incorporated here.

5. Data valuation (scoreval_checksel.py) - Assigns scores to training datapoints using selected checkpoints/trajectories.

6. Subset selection (topsel.py or diverselect.py) - Returns a subset of datapoints as (a) top k elements or by (b) executing SimSel algorithm using the assigned scores. 


# Using Checksel for other datasets/models

<b> Change in dataset </b>

For changing datasets, one needs to modify the definitions of the classes ```Dataset_train``` and ```Dataset_test``` in ```dataset.py``` as per requirement.

<b> Change in model </b>

1. Current ResNet implementation is a modified form of the original ResNet suited for images of lower dimension like that of CIFAR10. If we want to use the different architectures of this version of ResNet, we can follow the comments in the code and simply change the self.block and self.layers in the _init_ of ```class Model``` in ```model.py```.

2. If we want to use the original ResNet18 architecture or any other existing architectures from torchvision.models, include the line 
```model = models.resnet18(pretrained=True) #use torch models``` in model.py.

3. ```get_grad()``` function in ```helper.py``` will have the last layer name changed as per the architecture. For the current model, the last layer name is ```self.linear``` ; hence for extracting the last layer output, line number 70 in ```helper.py``` is ```params = model.linear.weight``` .

For the inbuilt torch model, the last layer name for ResNet implementations, is ```self.fc``` ; hence for extracting the last layer output, the line will turn to ```params = model.fc.weight```.

4. ```scoreval_checksel``` requires a method to extract features of an image to compute neighbours. This will change as per the architecture to be used, i.e one needs to know from which layer the feature needs to be extracted. In this case, if we want to extract the feature from the penultimate layer, we need to know its name which for this case is avgpool, so we define ```self.layer = self.model._modules.get('avgpool')```.

Overall, while using any other torch models,  one needs to definitely know:

1. the last layer name for computing gradients.

2. the layer name from where feature has to be extracted.

3. the feature dimension of extracted features, which can be specified beforehand in config_create.py under the key ‘featuredim’.

<b> Change in training parameters </b>

1. For learning rate, optimizer, one needs to change them from ```def _init_ ```of ```class TrajSel()``` in ```run.py```; 

2. For type of loss like CrossEntropy, one needs to specify them in ```def _init_ ```of ```class TrajSel()``` in ```run.py``` and ```def _init_ ```of ```class HelperFunc()``` in ```helper.py```.
