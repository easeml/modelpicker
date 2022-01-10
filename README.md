# Data Labeling for Testing and Model Selection
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

This repository aims to provide you the state-of-the-art algorithm that performs active data labeling for testing and model selection, the so-called [```modelpicker```](http://proceedings.mlr.press/v130/reza-karimi21a/reza-karimi21a.pdf). The ```modelpicker``` algorithm queries the labels of the most informative data instances such that the model with the highest test accuracy for your target prediction task in hand can be found with a minimal labeling cost. 

## Overview
Below we provide an overview of ```modelpicker```. For a quick start, please check [here](https://github.com/easeml/modelpicker#usage) or [our example notebook](https://github.com/easeml/modelpicker/blob/main/example.ipynb).


### Use cases
There are several scenarios where you can employ ```modelpicker```. The leading use cases are as follows:
- Imagine you have many pretrained models that are trained on different data slices. For a freshly collected new dataset for which you would like to make predictions, you may not want to train a model from stratch but rather select the pretrained one with maximum generalization accuracy on this fresh dataset. The ```modelpicker``` can achieve this with a minimal labeling cost. This has crucial importance in cases where there is a significant data drift compared to the training distribution in which your most recent model is trained on, and retraining process is inefficient and moreover unnecessary due to the availability of a variety of previously trained models.
- The ```modelpicker``` immediately applies to all scenarios in which you would like to select the model with the highest generalization accuracy on the your target task. The term "model" here refer to any distinction between classifiers ranging from the training/validation sets they are trained on to architectures or entirely different ML models.

### Principle
The ```modelpicker``` scans the data, and it makes a random query decision upon seeing each instance. This random decision is simply a coin flip with an adaptive bias. If it comes heads, ```modelpicker``` queries the label for this instance, else it does not. At each round, the bias is computed using the evidence on the previously labelled instances as well as the disagreement this individual instance creates among the pretrained models. Below is a general overview of the coin flipping principle. We refer to our paper [here](http://proceedings.mlr.press/v130/reza-karimi21a/reza-karimi21a.pdf) for further details.

<p align="center">
  <img src="modelpicker.png" alt="modelpicker" width="800"/>
</p>
<p align="center">
    <em>Modelpicker is a biased coin flipping strategy where at each round, the bias is computed based on the partial evidence and disagreement of the instance among the models. Upon exceeding a labeling budget specified by the user, the modelpicker returns the model that it beliefs to be the best.</em>
</p>



### Why modelpicker?
```modelpicker``` is a strategy that speciliazes on selecting most informative instances with a mere aim to find the best pretrained model. Yet there are several other strategies to select the most informative instances although they have different objectives than that of modelpicker. Despite that, these active and/or online learning strategies are in general very competitive baselines measuring/sorting the uncertainty of data instances. We adapted those strategies for model selection and performed exhaustive comparisons to modelpicker, in which we observed a significant benefit in using ```modelpicker```. Below results illustrate a summary of the comparison. For other evaluation metrices other than success probabilities, such as regret and accuracy gaps, we refer to [here](https://github.com/DS3Lab/online-active-model-selection) and [here](http://proceedings.mlr.press/v130/reza-karimi21a/reza-karimi21a.pdf) for a detailed look.

<p align="center">
  <img src="comparison.png" alt="comparison" width="1000"/>
</p>
<p align="center">
    <em>The success probabilities (probability of outputting the true best model) indicate that modelpicker has a significant improvement over other selective sampling baselines and consistently over different datasets. Note that it takes only 12% of labeling effort for modelpicker to output the best model confidently on ImageNet dataset!</em>
</p>


## Usage
To run experiment on a set of collected pre-trained models, run this command:

```buildoutcfg
python3 modelpicker.py [--predictions] [--labelset] [--budget]
```

```buildoutcfg
arguments:
--predictions PREDICTIONS 
                          The name of your CSV file consisting of model predictions. This is a 2D array of 
                          model predictions on your freshly collected data with size 𝑁×𝑘 where 𝑁 is the 
                          amount of unlabeled instances available at time 𝑡, and 𝑘 is the number of models. 
                          Each prediction is mapped to an integer.
--labelspace LABELSPACE
                          The name of your CSV file consisting of elements of label space. For instance, for a dataset consisting 
                          of 4 classes, a possible label space can be {0,1,2,3}. These labels should be consistent 
                          with the mapping used for prediction matrix as well.
--budget BUDGET 
                          An integer that indicates the labeling budget

outputs:
--bestmodel  
                          The best model based on the queried labels 
--beliefs 
                          The posterior belief on the models being best.
```
### Example
Using the emotion detection task and predictions of pretrained models in ```data/```, we can run the following command to label 10 instances to find out the best model for this task, which will in turn be used to make predictions on the remaining unlabelled instances.
```buildoutcfg
python3 modelpicker.py data/emocontext/predictions data/emocontext/labelspace 10
```

A jupyter notebook [`example.ipynb`](https://github.com/easeml/modelpicker/blob/main/example.ipynb) is available in the main repository to illustrate how to use the code with the arguments. 

## Citations

```bibtex
% Algorithm and Theory 
@article{karimigurel2021mp,
  title={Online Active Model Selection for Pretrained Classifiers},
  author={Karimi, Mohammad Reza and Gurel, Nezihe Merve and Karlas, Bojan and Rausch, Johannes and Zhang, Ce and Krause, Andreas},
  journal={International Conference on Artificial Intelligence and Statistics},
  volume={130},
  year={2021}
}
