# RobOT: Robustness-Oriented Testing for Deep Learning Systems published at ICSE 2021
See the <a href="https://arxiv.org/pdf/2102.05913.pdf" target="_blank">ICSE2021 paper</a>  for more details. 

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

then start jupyter notebook for experiments
```
jupyter notebook
```

## Files
- MNIST - robustness experiments on the MNIST dataset.
- FASHION - robustness experimnets on the FASHION dataset.
- SVHN - robustness experiments on the SVHN dataset.
- CIFAR-10 - robustness experiments on the CIFAR-10 dataset.



## Functions
metrics.py contains proposed metrics FOL. 

train_model.py is to train the DNN model.

attack.py contains FGSM and PGD attack. 

gen_adv.py is to generate adversarial inputs for test selection and robustness evaluation. You could also use toolbox like <a href="https://github.com/cleverhans-lab/cleverhans" target="_blank">cleverhans</a> for the test case generation. 

select_retrain.py is to select valuable test cases for model retraining. 


For testing methods (DeepXplore, DLFuzz, ADAPT), we use the code repository <a href="https://github.com/kupl/ADAPT" target="_blank">ADAPT</a>. 

For testing methods (AEQUITAS, ADF), we use the code repository <a href="https://github.com/pxzhang94/ADF" target="_blank">ADF</a>. 

## Coming soon
More details would be included soon. 





