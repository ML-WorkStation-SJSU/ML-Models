# ML-Models
Study of machine learning models

* [Random Forest](https://github.com/ML-WorkStation-SJSU/ML-Models/blob/master/Random%20Forest/JoyHome/Random%20Forest.ipynb)
* [LDA]() **(if time is enough, and QDA)**
* [SVM]()
* [HMM]()
* [PCA]()
* [Naive Bayes]()
* [Guassian Process]()
* [AdaBoost]()
* [XgBoost]()
* [Logistic Regression](https://towardsdatascience.com/real-world-implementation-of-logistic-regression-5136cefb8125)

# Related Concepts

* [Maximum Likehood Estimation]()
* [EM Algotithm]()

1. Algorithm Descriptions

---
Here is an overview of the linear, nonlinear and ensemble algorithm descriptions:

- Algorithm 1: Gradient Descent.
- Algorithm 2: Linear Regression.
- Algorithm 3: Logistic Regression.
- Algorithm 4: Linear Discriminant Analysis.
- Algorithm 5: Classification and Regression Trees.
- Algorithm 6: Naive Bayes.
- Algorithm 7: K-Nearest Neighbors.
- Algorithm 8: Learning Vector Quantization.
- Algorithm 9: Support Vector Machines.
- Algorithm 10: Bagged Decision Trees and Random Forest.
- Algorithm 11: Boosting and AdaBoost.

2. Algorithm Tutorials

---
Here is an overview of the step-by-step algorithm tutorials:

- Tutorial 1: Simple Linear Regression using Statistics.
- Tutorial 2: Simple Linear Regression with Gradient Descent.
- Tutorial 3: Logistic Regression with Gradient Descent.
- Tutorial 4: Linear Discriminant Analysis using Statistics.
- Tutorial 5: Classification and Regression Trees with Gini.
- Tutorial 6: Naive Bayes for Categorical Data.
- Tutorial 7: Gaussian Naive Bayes for Real-Valued Data.
- Tutorial 8: K-Nearest Neighbors for Classification.
- Tutorial 9: Learning Vector Quantization for Classification.
- Tutorial 10: Support Vector Machines with Gradient Descent.
- Tutorial 11: Bagged Classification and Regression Trees.
- Tutorial 12: AdaBoost for Classification.

<div class="section" id="supervised-learning">
<span id="id1"></span><h1>1. Supervised learning<a class="headerlink" href="#supervised-learning" title="Permalink to this headline">¶</a></h1>
<div class="toctree-wrapper compound">
<ul>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html">1.1. Linear Models</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares">1.1.1. Ordinary Least Squares</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification">1.1.2. Ridge regression and classification</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#lasso">1.1.3. Lasso</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#multi-task-lasso">1.1.4. Multi-task Lasso</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#elastic-net">1.1.5. Elastic-Net</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net">1.1.6. Multi-task Elastic-Net</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression">1.1.7. Least Angle Regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#lars-lasso">1.1.8. LARS Lasso</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp">1.1.9. Orthogonal Matching Pursuit (OMP)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression">1.1.10. Bayesian Regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression">1.1.11. Logistic regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd">1.1.12. Stochastic Gradient Descent - SGD</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#perceptron">1.1.13. Perceptron</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms">1.1.14. Passive Aggressive Algorithms</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors">1.1.15. Robustness regression: outliers and modeling errors</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions">1.1.16. Polynomial regression: extending linear models with basis functions</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/lda_qda.html">1.2. Linear and Quadratic Discriminant Analysis</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/lda_qda.html#dimensionality-reduction-using-linear-discriminant-analysis">1.2.1. Dimensionality reduction using Linear Discriminant Analysis</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-the-lda-and-qda-classifiers">1.2.2. Mathematical formulation of the LDA and QDA classifiers</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-lda-dimensionality-reduction">1.2.3. Mathematical formulation of LDA dimensionality reduction</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/lda_qda.html#shrinkage">1.2.4. Shrinkage</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/lda_qda.html#estimation-algorithms">1.2.5. Estimation algorithms</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/kernel_ridge.html">1.3. Kernel ridge regression</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html">1.4. Support Vector Machines</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#classification">1.4.1. Classification</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#regression">1.4.2. Regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#density-estimation-novelty-detection">1.4.3. Density estimation, novelty detection</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#complexity">1.4.4. Complexity</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use">1.4.5. Tips on Practical Use</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#kernel-functions">1.4.6. Kernel functions</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation">1.4.7. Mathematical formulation</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/svm.html#implementation-details">1.4.8. Implementation details</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html">1.5. Stochastic Gradient Descent</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#classification">1.5.1. Classification</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#regression">1.5.2. Regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#stochastic-gradient-descent-for-sparse-data">1.5.3. Stochastic Gradient Descent for sparse data</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#complexity">1.5.4. Complexity</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#stopping-criterion">1.5.5. Stopping criterion</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use">1.5.6. Tips on Practical Use</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation">1.5.7. Mathematical formulation</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/sgd.html#implementation-details">1.5.8. Implementation details</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html">1.6. Nearest Neighbors</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-nearest-neighbors">1.6.1. Unsupervised Nearest Neighbors</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification">1.6.2. Nearest Neighbors Classification</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression">1.6.3. Nearest Neighbors Regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms">1.6.4. Nearest Neighbor Algorithms</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html#nearest-centroid-classifier">1.6.5. Nearest Centroid Classifier</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-transformer">1.6.6. Nearest Neighbors Transformer</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neighbors.html#neighborhood-components-analysis">1.6.7. Neighborhood Components Analysis</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/gaussian_process.html">1.7. Gaussian Processes</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr">1.7.1. Gaussian Process Regression (GPR)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/gaussian_process.html#gpr-examples">1.7.2. GPR examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc">1.7.3. Gaussian Process Classification (GPC)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/gaussian_process.html#gpc-examples">1.7.4. GPC examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes">1.7.5. Kernels for Gaussian Processes</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/cross_decomposition.html">1.8. Cross decomposition</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/naive_bayes.html">1.9. Naive Bayes</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes">1.9.1. Gaussian Naive Bayes</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes">1.9.2. Multinomial Naive Bayes</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes">1.9.3. Complement Naive Bayes</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes">1.9.4. Bernoulli Naive Bayes</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes">1.9.5. Categorical Naive Bayes</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/naive_bayes.html#out-of-core-naive-bayes-model-fitting">1.9.6. Out-of-core naive Bayes model fitting</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html">1.10. Decision Trees</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#classification">1.10.1. Classification</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#regression">1.10.2. Regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#multi-output-problems">1.10.3. Multi-output problems</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#complexity">1.10.4. Complexity</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use">1.10.5. Tips on practical use</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart">1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation">1.10.7. Mathematical formulation</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning">1.10.8. Minimal Cost-Complexity Pruning</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html">1.11. Ensemble methods</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator">1.11.1. Bagging meta-estimator</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees">1.11.2. Forests of randomized trees</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#adaboost">1.11.3. AdaBoost</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting">1.11.4. Gradient Tree Boosting</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting">1.11.5. Histogram-Based Gradient Boosting</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier">1.11.6. Voting Classifier</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor">1.11.7. Voting Regressor</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization">1.11.8. Stacked generalization</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html">1.12. Multiclass and multilabel algorithms</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification-format">1.12.1. Multilabel classification format</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#one-vs-the-rest">1.12.2. One-Vs-The-Rest</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#one-vs-one">1.12.3. One-Vs-One</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#error-correcting-output-codes">1.12.4. Error-Correcting Output-Codes</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression">1.12.5. Multioutput regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#multioutput-classification">1.12.6. Multioutput classification</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#classifier-chain">1.12.7. Classifier Chain</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/multiclass.html#regressor-chain">1.12.8. Regressor Chain</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/feature_selection.html">1.13. Feature selection</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance">1.13.1. Removing features with low variance</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection">1.13.2. Univariate feature selection</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination">1.13.3. Recursive feature elimination</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel">1.13.4. Feature selection using SelectFromModel</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-as-part-of-a-pipeline">1.13.5. Feature selection as part of a pipeline</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/label_propagation.html">1.14. Semi-Supervised</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/label_propagation.html#label-propagation">1.14.1. Label Propagation</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/isotonic.html">1.15. Isotonic regression</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/calibration.html">1.16. Probability calibration</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html">1.17. Neural network models (supervised)</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron">1.17.1. Multi-layer Perceptron</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification">1.17.2. Classification</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression">1.17.3. Regression</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regularization">1.17.4. Regularization</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#algorithms">1.17.5. Algorithms</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#complexity">1.17.6. Complexity</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#mathematical-formulation">1.17.7. Mathematical formulation</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use">1.17.8. Tips on Practical Use</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html#more-control-with-warm-start">1.17.9. More control with warm_start</a></li>
</ul>
</li>
</ul>
</div>
</div>

<div class="section" id="unsupervised-learning">
<span id="id1"></span><h1>2. Unsupervised learning<a class="headerlink" href="#unsupervised-learning" title="Permalink to this headline">¶</a></h1>
<div class="toctree-wrapper compound">
<ul>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/mixture.html">2.1. Gaussian mixture models</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture">2.1.1. Gaussian Mixture</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/mixture.html#variational-bayesian-gaussian-mixture">2.1.2. Variational Bayesian Gaussian Mixture</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html">2.2. Manifold learning</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#introduction">2.2.1. Introduction</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#isomap">2.2.2. Isomap</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding">2.2.3. Locally Linear Embedding</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#modified-locally-linear-embedding">2.2.4. Modified Locally Linear Embedding</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#hessian-eigenmapping">2.2.5. Hessian Eigenmapping</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#spectral-embedding">2.2.6. Spectral Embedding</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#local-tangent-space-alignment">2.2.7. Local Tangent Space Alignment</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds">2.2.8. Multi-dimensional Scaling (MDS)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne">2.2.9. t-distributed Stochastic Neighbor Embedding (t-SNE)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/manifold.html#tips-on-practical-use">2.2.10. Tips on practical use</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html">2.3. Clustering</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods">2.3.1. Overview of clustering methods</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#k-means">2.3.2. K-means</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation">2.3.3. Affinity Propagation</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#mean-shift">2.3.4. Mean Shift</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering">2.3.5. Spectral clustering</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering">2.3.6. Hierarchical clustering</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#dbscan">2.3.7. DBSCAN</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#optics">2.3.8. OPTICS</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#birch">2.3.9. Birch</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation">2.3.10. Clustering performance evaluation</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/biclustering.html">2.4. Biclustering</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/biclustering.html#spectral-co-clustering">2.4.1. Spectral Co-Clustering</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/biclustering.html#spectral-biclustering">2.4.2. Spectral Biclustering</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/biclustering.html#biclustering-evaluation">2.4.3. Biclustering evaluation</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html">2.5. Decomposing signals in components (matrix factorization problems)</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca">2.5.1. Principal component analysis (PCA)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis">2.5.2. Truncated singular value decomposition and latent semantic analysis</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html#dictionary-learning">2.5.3. Dictionary Learning</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html#factor-analysis">2.5.4. Factor Analysis</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html#independent-component-analysis-ica">2.5.5. Independent component analysis (ICA)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf">2.5.6. Non-negative matrix factorization (NMF or NNMF)</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda">2.5.7. Latent Dirichlet Allocation (LDA)</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/covariance.html">2.6. Covariance estimation</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/covariance.html#empirical-covariance">2.6.1. Empirical covariance</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance">2.6.2. Shrunk Covariance</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/covariance.html#sparse-inverse-covariance">2.6.3. Sparse inverse covariance</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/covariance.html#robust-covariance-estimation">2.6.4. Robust Covariance Estimation</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/outlier_detection.html">2.7. Novelty and Outlier Detection</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/outlier_detection.html#overview-of-outlier-detection-methods">2.7.1. Overview of outlier detection methods</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection">2.7.2. Novelty Detection</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/outlier_detection.html#id1">2.7.3. Outlier Detection</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection-with-local-outlier-factor">2.7.4. Novelty detection with Local Outlier Factor</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/density.html">2.8. Density Estimation</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/density.html#density-estimation-histograms">2.8.1. Density Estimation: Histograms</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation">2.8.2. Kernel Density Estimation</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html">2.9. Neural network models (unsupervised)</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html#restricted-boltzmann-machines">2.9.1. Restricted Boltzmann machines</a></li>
</ul>
</li>
</ul>
</div>
</div>
