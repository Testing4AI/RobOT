# RobOT: Robustness-Oriented Testing for DL models
See the ICSE2021 paper for more details. 


## Prerequisite (Py3.6 & Tf2)
The code are run successfully using Python 3.6 and Tensorflow 2.2.0.

We recommend using conda to install the tensorflow-gpu environment
```shell
conda create -n tf2-gpu tensorflow-gpu==2.2.0
conda activate tf2-gpu
```

Checking installed environments
```shell
conda env list
```

to run the code in jupyter, you should add the kernel in jupyter notebook 
```
pip install ipykernel
python -m ipykernel install --name tf2-gpu
```

then start jupyter process for some experiments
```
jupyter notebook
```


## File structure
- MNIST - experiments for MNIST dataset.
- Fashion - experiments for Fashion dataset.
- SVHN - experimnets for SVHN dataset.
- Cifar - experimnets for CIFAR-10 dataset.


## Coming soon
More details would be included soon. 





