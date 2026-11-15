import { BaseClassifier } from './base.js';
import { dot, norm } from '../utils/math.js';

export class LinearSVM extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.C = options.C ?? 1.0;
    this.lr = options.lr ?? 0.05;
    this.epochs = options.epochs ?? 80;
    this.w = [0, 0];
    this.b = 0;
    this.supportVectorIndices = [];
  }

  predictRaw(x) {
    return dot(this.w, x) + this.b;
  }

  predict(x) {
    return this.predictRaw(x) >= 0 ? 1 : -1;
  }

  predictProba(x) {
    const raw = this.predictRaw(x);
    // Sigmoid mapping of margin distance for heatmap
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, raw * 1.5))));
  }

  *fitSteps(X, y) {
    const n = X.length;
    const d = X[0].length;
    this.w = new Array(d).fill(0);
    this.b = 0;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let loss = 0;
      const svIndices = [];

      for (let i = 0; i < n; i++) {
        const margin = y[i] * this.predictRaw(X[i]);
        if (margin <= 1.0) {
          svIndices.push(i);
          loss += (1 - margin);
          // Subgradient update
          for (let j = 0; j < d; j++) {
            this.w[j] = this.w[j] * (1 - this.lr * 0.01) + this.lr * this.C * y[i] * X[i][j];
          }
          this.b += this.lr * this.C * y[i];
        } else {
          for (let j = 0; j < d; j++) {
            this.w[j] *= (1 - this.lr * 0.01);
          }
        }
      }

      this.supportVectorIndices = svIndices;
      const marginWidth = norm(this.w) > 1e-6 ? 2.0 / norm(this.w) : 0;

      yield {
        type: 'step',
        classifier: 'LinearSVM',
        step: epoch + 1,
        epoch: epoch + 1,
        w: [...this.w],
        b: this.b,
        marginWidth,
        supportVectorIndices: [...svIndices],
        loss: loss / n,
        accuracy: this.calculateAccuracy(X, y)
      };
    }

    this.isFitted = true;
    yield {
      type: 'complete',
      classifier: 'LinearSVM',
      step: this.epochs,
      w: [...this.w],
      b: this.b,
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

