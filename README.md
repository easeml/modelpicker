# Active Model Selection for Pretrained Classifiers
This repository aims to provide you the best pretrained model for your target distribution by keeping a minimal labeling cost. See how you can use it below.

## Usage
To run experiment on a set of collected pre-trained models, run this command:

```buildoutcfg
python3 -m modelpicker [--predictions ] [--labelset] [--budget]
```

```buildoutcfg
arguments:
--predictions PREDICTIONS 
                          A numpy array of model predictions. This is a matrix of 𝑁×𝑘 predictions where 𝑁 is the 
                          amount of unlabeled instances available at time 𝑡, and 𝑘 is the number of models. Each 
                          prediction is mapped to an integer.
--labelset LABELSET 
                          A numpy array consisting of label set elements. For instance, for a dataset consisting of 4 
                          classes, a possible label set can be {0,1,2,3}. These labels should be consistent with the 
                          mapping used for prediction matrix as well.
--budget BUDGET 
                          AAn integer that indicates the labeling budget

outputs:
--bestmodel  
                          The best model based on the queried labels 
--beliefs 
                          The posterior belief on the models being best.
```
## Example
A jupyter notebook `example.ipynb` is available in the main repository to illustrate how to use the code with the arguments. 

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
