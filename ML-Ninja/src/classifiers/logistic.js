import { BaseClassifier } from './base.js';
import { dot, sigmoid } from '../utils/math.js';

export class LogisticRegression extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.lr = options.lr ?? 0.2;
    this.epochs = options.epochs ?? 80;
    this.w = [0, 0];
    this.b = 0;
    this.lossHistory = [];
  }

  predictRaw(x) {
    return dot(this.w, x) + this.b;
  }

  predictProba(x) {
    return sigmoid(this.predictRaw(x));
  }

  predict(x) {
    return this.predictProba(x) >= 0.5 ? 1 : -1;
  }

  *fitSteps(X, y) {
    const n = X.length;
    const d = X[0].length;
    this.w = new Array(d).fill(0);
    this.b = 0;
    this.lossHistory = [];

    // Convert labels from {-1, 1} to {0, 1}
    const t = y.map(val => (val >= 0 ? 1 : 0));

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let gradW = new Array(d).fill(0);
      let gradB = 0;
      let totalLoss = 0;

      for (let i = 0; i < n; i++) {
        const prob = sigmoid(this.predictRaw(X[i]));
        const err = prob - t[i];

        for (let j = 0; j < d; j++) {
          gradW[j] += err * X[i][j];
        }
        gradB += err;

        // Binary cross entropy
        const pClamped = Math.max(1e-12, Math.min(1 - 1e-12, prob));
        totalLoss += -(t[i] * Math.log(pClamped) + (1 - t[i]) * Math.log(1 - pClamped));
      }

      for (let j = 0; j < d; j++) {
        this.w[j] -= (this.lr * gradW[j]) / n;
      }
      this.b -= (this.lr * gradB) / n;

      const loss = totalLoss / n;
      this.lossHistory.push(loss);

      yield {
        type: 'step',
        classifier: 'LogisticRegression',
        step: epoch + 1,
        epoch: epoch + 1,
        w: [...this.w],
        b: this.b,
        loss,
        lossHistory: [...this.lossHistory],
        accuracy: this.calculateAccuracy(X, y)
      };
    }

    this.isFitted = true;
    yield {
      type: 'complete',
      classifier: 'LogisticRegression',
      step: this.epochs,
      w: [...this.w],
      b: this.b,
      loss: this.lossHistory[this.lossHistory.length - 1],
      lossHistory: [...this.lossHistory],
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

