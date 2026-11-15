import { Perceptron } from './perceptron.js';
import { LogisticRegression } from './logistic.js';
import { LinearSVM } from './svm.js';
import { KernelSVM } from './kernelSvm.js';
import { KNN } from './knn.js';
import { GaussianNaiveBayes } from './naiveBayes.js';
import { DiscriminantAnalysis } from './lda.js';
import { DecisionTree } from './tree.js';
import { RandomForest } from './forest.js';
import { MLP } from './mlp.js';

export const CLASSIFIERS = {
  perceptron: {
    id: 'perceptron',
    name: 'Perceptron',
    Class: Perceptron,
    formula: 'w · x + b = 0',
    summary: 'The simplest linear binary classifier. Iteratively updates its weight vector whenever it encounters a misclassified training sample.',
    whenItFails: 'Fails to converge when data is not linearly separable (e.g. XOR, concentric circles, interleaved moons).',
    defaultParams: { lr: 0.1, epochs: 40 }
  },
  logistic: {
    id: 'logistic',
    name: 'Logistic Regression',
    Class: LogisticRegression,
    formula: 'P(y=1|x) = 1 / (1 + e^{-(w·x + b)})',
    summary: 'Models the posterior class probability using a logistic sigmoid link function, trained via gradient descent on binary cross-entropy loss.',
    whenItFails: 'Strictly linear decision boundary; unable to separate non-linear distributions like XOR without feature transforms.',
    defaultParams: { lr: 0.25, epochs: 70 }
  },
  svm: {
    id: 'svm',
    name: 'Linear SVM',
    Class: LinearSVM,
    formula: 'min 1/2||w||² + C ∑ max(0, 1 - y_i(w·x_i + b))',
    summary: 'Finds the optimal separating hyperplane that maximizes the geometric margin between classes, guided by the closest support vectors.',
    whenItFails: 'Fails on non-linearly separable distributions when restricted to a linear kernel.',
    defaultParams: { C: 1.0, lr: 0.05, epochs: 60 }
  },
  kernelSvm: {
    id: 'kernelSvm',
    name: 'Kernel SVM (RBF)',
    Class: KernelSVM,
    formula: 'K(x, x\') = exp(-γ ||x - x\'||²)',
    summary: 'Implicitly projects 2D features into an infinite-dimensional Hilbert space via the RBF kernel to draw flexible non-linear margins.',
    whenItFails: 'Prone to severe overfitting if gamma (γ) or C is set excessively high, memorizing individual noise points.',
    defaultParams: { kernel: 'rbf', gamma: 0.6, C: 2.5, lr: 0.06, epochs: 50 }
  },
  knn: {
    id: 'knn',
    name: 'k-Nearest Neighbors',
    Class: KNN,
    formula: 'y_pred = mode({y_i : x_i ∈ N_k(x)})',
    summary: 'Non-parametric instance-based lazy learner. Classifies new query points by taking a majority vote among its k closest neighbors in feature space.',
    whenItFails: 'High sensitivity to noise and outliers when k=1; computationally expensive inference on large datasets; metric distortion.',
    defaultParams: { k: 5, metric: 'euclidean' }
  },
  naiveBayes: {
    id: 'naiveBayes',
    name: 'Gaussian Naive Bayes',
    Class: GaussianNaiveBayes,
    formula: 'P(c|x) ∝ P(c) ∏ N(x_j; μ_cj, σ²_cj)',
    summary: 'Probabilistic generative model assuming all features are conditionally independent given the class label. Fits axis-aligned Gaussian distributions.',
    whenItFails: 'Breaks down when features have strong correlations (fails to rotate covariance ellipses) or multi-modal clusters.',
    defaultParams: {}
  },
  lda: {
    id: 'lda',
    name: 'Linear Discriminant Analysis',
    Class: DiscriminantAnalysis,
    formula: 'δ_k(x) = xᵀ Σ⁻¹ μ_k - 1/2 μ_kᵀ Σ⁻¹ μ_k + log π_k',
    summary: 'Models classes as multivariate Gaussians with identical pooled covariance matrix Σ, resulting in an optimal linear decision surface.',
    whenItFails: 'Fails when classes have drastically unequal variances or non-linear separation curves (switch to QDA mode).',
    defaultParams: { mode: 'lda' }
  },
  qda: {
    id: 'qda',
    name: 'Quadratic Discriminant Analysis',
    Class: DiscriminantAnalysis,
    formula: 'δ_k(x) = -1/2 log|Σ_k| - 1/2(x-μ_k)ᵀ Σ_k⁻¹(x-μ_k) + log π_k',
    summary: 'Generative Gaussian classifier allowing distinct covariance matrices for each class, yielding quadratic curved boundaries.',
    whenItFails: 'Requires more data points to estimate covariance matrices reliably; can overfit with small sample sizes.',
    defaultParams: { mode: 'qda' }
  },
  tree: {
    id: 'tree',
    name: 'Decision Tree (CART)',
    Class: DecisionTree,
    formula: 'Gini = 1 - ∑ p_i², split argmax ΔGini',
    summary: 'Recursively partitions feature space using orthogonal axis-aligned splits that maximize Gini impurity reduction or information gain.',
    whenItFails: 'High variance and instability; diagonal decision boundaries require deep stairstep approximations.',
    defaultParams: { maxDepth: 4, minSamplesSplit: 2, criterion: 'gini' }
  },
  forest: {
    id: 'forest',
    name: 'Random Forest',
    Class: RandomForest,
    formula: 'y_pred = 1/B ∑ Tree_b(x)',
    summary: 'Ensemble of decorrelated decision trees trained on bootstrapped subsets of the data. Reduces variance through democratic majority voting.',
    whenItFails: 'Slightly slower inference and loss of tree interpretability compared to a single decision tree.',
    defaultParams: { nTrees: 6, maxDepth: 4 }
  },
  mlp: {
    id: 'mlp',
    name: 'Multi-Layer Perceptron',
    Class: MLP,
    formula: 'y = σ(W₂ · tanh(W₁ x + b₁) + b₂)',
    summary: 'Artificial neural network with a non-linear hidden layer that learns internal distributed representations via backpropagation gradient descent.',
    whenItFails: 'Susceptible to getting trapped in local minima or slow convergence if learning rate is improperly tuned.',
    defaultParams: { hiddenDim: 8, lr: 0.15, epochs: 80 }
  }
};

