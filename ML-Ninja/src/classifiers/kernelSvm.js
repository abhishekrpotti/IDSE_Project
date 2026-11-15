import { BaseClassifier } from './base.js';
import { dot, euclideanDistance } from '../utils/math.js';

export class KernelSVM extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.kernelType = options.kernel ?? 'rbf'; // 'rbf' | 'poly'
    this.gamma = options.gamma ?? 0.5;
    this.degree = options.degree ?? 3;
    this.C = options.C ?? 2.0;
    this.lr = options.lr ?? 0.05;
    this.epochs = options.epochs ?? 60;
    this.alpha = [];
    this.b = 0;
    this.X = [];
    this.y = [];
    this.supportVectorIndices = [];
  }

  kernel(x1, x2) {
    if (this.kernelType === 'poly') {
      return Math.pow(dot(x1, x2) * 0.1 + 1, this.degree);
    }
    // Default: RBF
    const dist = euclideanDistance(x1, x2);
    return Math.exp(-this.gamma * dist * dist);
  }

  predictRaw(x) {
    let sum = this.b;
    for (let i = 0; i < this.X.length; i++) {
      if (Math.abs(this.alpha[i]) > 1e-4) {
        sum += this.alpha[i] * this.y[i] * this.kernel(this.X[i], x);
      }
    }
    return sum;
  }

  predict(x) {
    return this.predictRaw(x) >= 0 ? 1 : -1;
  }

  predictProba(x) {
    const raw = this.predictRaw(x);
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, raw * 1.5))));
  }

  *fitSteps(X, y) {
    const n = X.length;
    this.X = X;
    this.y = y;
    this.alpha = new Array(n).fill(0);
    this.b = 0;

    // Precompute kernel matrix K[i][j]
    const K = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        K[i][j] = this.kernel(X[i], X[j]);
      }
    }

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const svIndices = [];

      for (let i = 0; i < n; i++) {
        // Compute current output f(x_i)
        let fi = this.b;
        for (let j = 0; j < n; j++) {
          if (this.alpha[j] > 1e-5) {
            fi += this.alpha[j] * y[j] * K[j][i];
          }
        }

        const margin = y[i] * fi;
        if (margin < 1.0) {
          this.alpha[i] = Math.min(this.C, this.alpha[i] + this.lr * (1.0 - margin));
          this.b += this.lr * y[i] * (1.0 - margin) * 0.5;
        } else {
          this.alpha[i] = Math.max(0, this.alpha[i] - this.lr * 0.05);
        }

        if (this.alpha[i] > 1e-4) {
          svIndices.push(i);
        }
      }

      this.supportVectorIndices = svIndices;

      yield {
        type: 'step',
        classifier: 'KernelSVM',
        step: epoch + 1,
        epoch: epoch + 1,
        supportVectorIndices: [...svIndices],
        numSupportVectors: svIndices.length,
        accuracy: this.calculateAccuracy(X, y)
      };
    }

    this.isFitted = true;
    yield {
      type: 'complete',
      classifier: 'KernelSVM',
      step: this.epochs,
      supportVectorIndices: [...this.supportVectorIndices],
      accuracy: this.calculateAccuracy(X, y)
    };
  }

  calculateAccuracy(X, y) {
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      if (this.predict(X[i]) === y[i]) correct++;
    }
    return correct / (X.length || 1);
  }
}

