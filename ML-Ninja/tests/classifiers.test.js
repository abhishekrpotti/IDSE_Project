import { describe, it, expect } from 'vitest';
import { generateDataset } from '../src/data/generators.js';
import { Perceptron } from '../src/classifiers/perceptron.js';
import { LogisticRegression } from '../src/classifiers/logistic.js';
import { LinearSVM } from '../src/classifiers/svm.js';
import { KernelSVM } from '../src/classifiers/kernelSvm.js';
import { KNN } from '../src/classifiers/knn.js';
import { GaussianNaiveBayes } from '../src/classifiers/naiveBayes.js';
import { DiscriminantAnalysis } from '../src/classifiers/lda.js';
import { DecisionTree } from '../src/classifiers/tree.js';
import { RandomForest } from '../src/classifiers/forest.js';
import { MLP } from '../src/classifiers/mlp.js';
import { dot, norm, invert2x2, det2x2, standardize } from '../src/utils/math.js';
import { calculateMetrics } from '../src/utils/metrics.js';

describe('Math & Metrics Utilities', () => {
  it('computes dot product and norm accurately', () => {
    expect(dot([1, 2], [3, 4])).toBe(11);
    expect(norm([3, 4])).toBe(5);
  });

  it('inverts 2x2 matrix correctly', () => {
    const M = [[2, 1], [1, 2]];
    const inv = invert2x2(M);
    expect(det2x2(M)).toBe(3);
    expect(inv[0][0]).toBeCloseTo(2 / 3);
    expect(inv[0][1]).toBeCloseTo(-1 / 3);
  });

  it('computes accuracy, precision, recall, and confusion matrix', () => {
    const yTrue = [1, 1, -1, -1];
    const yPred = [1, -1, -1, 1];
    const metrics = calculateMetrics(yTrue, yPred);
    expect(metrics.accuracy).toBe(0.5);
    expect(metrics.tp).toBe(1);
    expect(metrics.fp).toBe(1);
    expect(metrics.tn).toBe(1);
    expect(metrics.fn).toBe(1);
  });
});

describe('Linear Classifiers on Separable Blobs', () => {
  const data = generateDataset('separable', { n: 40, noise: 0.1, seed: 42 });

  it('Perceptron achieves 100% accuracy on linearly separable data', () => {
    const model = new Perceptron({ lr: 0.1, epochs: 50 });
    model.fit(data.X, data.y);
    const acc = model.calculateAccuracy(data.X, data.y);
    expect(acc).toBe(1.0);
  });

  it('Logistic Regression achieves > 95% accuracy on separable data', () => {
    const model = new LogisticRegression({ lr: 0.3, epochs: 80 });
    model.fit(data.X, data.y);
    const acc = model.calculateAccuracy(data.X, data.y);
    expect(acc).toBeGreaterThanOrEqual(0.95);
  });

  it('Linear SVM achieves > 95% accuracy on separable data', () => {
    const model = new LinearSVM({ C: 1.0, lr: 0.05, epochs: 60 });
    model.fit(data.X, data.y);
    const acc = model.calculateAccuracy(data.X, data.y);
    expect(acc).toBeGreaterThanOrEqual(0.95);
  });

  it('LDA achieves > 95% accuracy on separable data', () => {
    const model = new DiscriminantAnalysis({ mode: 'lda' });
    model.fit(data.X, data.y);
    const acc = model.calculateAccuracy(data.X, data.y);
    expect(acc).toBeGreaterThanOrEqual(0.95);
  });

  it('Gaussian Naive Bayes achieves > 95% accuracy on separable data', () => {
    const model = new GaussianNaiveBayes();
    model.fit(data.X, data.y);
    const acc = model.calculateAccuracy(data.X, data.y);
    expect(acc).toBeGreaterThanOrEqual(0.95);
  });
});

describe('Non-Linear Classifiers on Complex Datasets', () => {
  const xorData = generateDataset('xor', { n: 60, noise: 0.1, seed: 123 });
  const circleData = generateDataset('circles', { n: 60, noise: 0.1, seed: 456 });

  it('Linear models fail on XOR while Decision Tree succeeds (> 85%)', () => {
    const perceptron = new Perceptron();
    perceptron.fit(xorData.X, xorData.y);
    const linearAcc = perceptron.calculateAccuracy(xorData.X, xorData.y);
    expect(linearAcc).toBeLessThanOrEqual(0.65); // Linear fails on XOR!

    const tree = new DecisionTree({ maxDepth: 4 });
    tree.fit(xorData.X, xorData.y);
    const treeAcc = tree.calculateAccuracy(xorData.X, xorData.y);
    expect(treeAcc).toBeGreaterThanOrEqual(0.85); // Tree succeeds on XOR!
  });

  it('Kernel SVM (RBF) solves concentric circles (> 90%)', () => {
    const svm = new KernelSVM({ kernel: 'rbf', gamma: 0.8, C: 3.0, epochs: 60 });
    svm.fit(circleData.X, circleData.y);
    const acc = svm.calculateAccuracy(circleData.X, circleData.y);
    expect(acc).toBeGreaterThanOrEqual(0.90);
  });

  it('k-NN solves concentric circles (> 90%)', () => {
    const knn = new KNN({ k: 3 });
    knn.fit(circleData.X, circleData.y);
    const acc = knn.calculateAccuracy(circleData.X, circleData.y);
    expect(acc).toBeGreaterThanOrEqual(0.90);
  });

  it('Random Forest solves XOR (> 85%)', () => {
    const forest = new RandomForest({ nTrees: 8, maxDepth: 4 });
    forest.fit(xorData.X, xorData.y);
    const acc = forest.calculateAccuracy(xorData.X, xorData.y);
    expect(acc).toBeGreaterThanOrEqual(0.85);
  });

  it('MLP neural network learns non-linear boundary on circles (> 85%)', () => {
    const mlp = new MLP({ hiddenDim: 8, lr: 0.18, epochs: 80 });
    mlp.fit(circleData.X, circleData.y);
    const acc = mlp.calculateAccuracy(circleData.X, circleData.y);
    expect(acc).toBeGreaterThanOrEqual(0.85);
  });
});

