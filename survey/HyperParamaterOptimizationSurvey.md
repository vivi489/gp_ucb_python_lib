# Hyper-paramter Optimization Survey

## Methods
- Grid Search 
- Random Search
- Sequential Model-based Algorithm Configuration (SMAC)
    - [Code](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/#software)
    - according to  https://arimo.com/data-science/2016/bayesian-optimization-hyperparameter-tuning/
        > SMAC  uses a random forest of regression trees to model the objective function, new points are sampled from the region considered optimal (high Expected Improvement) by the random forest
        
    
     
- Sequential Model-Based Optimization (SMBO) : BaysianOptimization
    - Tree-Structered Parzen Estimator (TPE)
        > TPE is an improved version of SMAC, where two separated models are used to model the posterior. 
        
        > The TPE algorithm is conspicuously deficient in optimizing each hyperparameter independently of the others. 
        It is almost certainly the case that the optimal values of some hyperparameters depend on settings of others. 
        Algorithms such as SMAC (Hutter et al., 2011) that can represent such interactions might be significantly more effective optimizers than TPE. 
        It might be possible to extend TPE to profitably employ non-factorial joint densities P(config|score).
    
    - Gaussian Process
    - Gaussian Markov Random Field (ours)
 



## Papers
- [Algorithms for Hyper-Parameter Optimization (NIPS 2011)](http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
    - Hyper paramter tuning for Deep learning 
    - Created "HyperOpt" library
    
- [Practical Bayesian Optimization of Machine Learning Algorithms (NIPS 2012)](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
    - Created "Spearmint" library
    - [Japanese slide](https://www.slideshare.net/utotch/practical-bayesian-optimization-of-machine-learning-algorithmsnips2012-24645254)

- [Bayesian Optimization with Robust Bayesian Neural Networks (NIPS 2016)](http://papers.nips.cc/paper/6117-bayesian-optimization-with-robust-bayesian-neural-networks.pdf)
    - [Presentation Movie](https://www.youtube.com/watch?v=F69q2ogaoBo)
    - Code is in RoBo library (Bohamiann)
    

- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization (ICLR2017)](https://openreview.net/pdf?id=ry18Ww5ee)
    - [Website](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html)
    - [Code](https://github.com/zygmuntz/hyperband)
    - According to [this Blog](http://fastml.com/tuning-hyperparams-fast-with-hyperband/)
        >  Hyperband runs configs for just an iteration or two at first, to get a taste of how they perform. Then it takes the best performers and runs them longer. Indeed, that’s all Hyperband does: run random configurations on a specific schedule of iterations per configuration, using earlier results to select candidates for longer runs.
    
- [Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets (AISTATS2017)](http://proceedings.mlr.press/v54/klein17a/klein17a.pdf)
    - Their method is "FABOLAS"
    - [Code](https://github.com/automl/RoBO/blob/master/robo/fmin/fabolas.py)

## Articles
- [Hyperoptなどのハイパーパラメータチューニングとその関連手法についてのメモ](http://paper.hatenadiary.jp/entry/2017/06/07/151158)
    
- [Hyperparameter optimization for Neural Networks](http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html)


## Libraries

[This site(AutoML)](http://www.ml4aad.org/automl/) is helpful.

### Hyper-parameter Optimization Libraries

#### SMAC3
The main core consists of Bayesian Optimization in combination with a simple racing mechanism to efficiently decide which of two configuration performs better.

#### HyperOpt
Python implementation of [Algorithms for Hyper-Parameter Optimization](http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)

Only tpe and random algorithms are supported.

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

```
#### Spearmint 
https://github.com/JasperSnoek/spearmint

### HPOlib: Library for testing your hyper-parameter optimization algorithm
This was introduced in the paper "Towards an Empirical Foundation for Assessing Bayesian Optimization of Hyperparameters".(NIPS2013 workshop) 

This library is used to test your hyperparameter optimizaition algorithm.

HPOlib didn't work but now HPOlib2 is available.


## References
- https://www.slideshare.net/hskksk/hyperopt
