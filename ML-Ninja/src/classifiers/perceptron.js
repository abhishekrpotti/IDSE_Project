import { BaseClassifier } from './base.js';
import { dot } from '../utils/math.js';

export class Perceptron extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.lr = options.lr ?? 0.1;
    this.maxEpochs = options.epochs ?? 50;
    this.w = [0, 0];
    this.b = 0;
  }

  predictRaw(x) {
    return dot(this.w, x) + this.b;
  }

  predict(x) {
    return this.predictRaw(x) >= 0 ? 1 : -1;
  }

  predictProba(x) {
    // Distance mapped through sigmoid for visual contour
    const z = this.predictRaw(x);
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z * 2.0))));
  }

  *fitSteps(X, y) {
    const d = X[0].length;
    this.w = new Array(d).fill(0);
    this.b = 0;
    let stepCount = 0;

    for (let epoch = 0; epoch < this.maxEpochs; epoch++) {
      let errors = 0;
      for (let i = 0; i < X.length; i++) {
        const xi = X[i];
        const yi = y[i];
        const raw = this.predictRaw(xi);
        const yPred = raw >= 0 ? 1 : -1;

        if (yPred !== yi) {
          errors++;
          // Update rule: w = w + lr * y * x
          for (let j = 0; j < d; j++) {
            this.w[j] += this.lr * yi * xi[j];
          }
          this.b += this.lr * yi;
          stepCount++;

          yield {
            type: 'step',
            classifier: 'Perceptron',
            step: stepCount,
            epoch,
            w: [...this.w],
            b: this.b,
            activePointIndex: i,
            activePoint: xi,
            activeLabel: yi,
            errors,
            accuracy: this.calculateAccuracy(X, y)
          };
        }
      }

      if (errors === 0) {
        break; // Converged
      }
    }

    this.isFitted = true;
    yield {
      type: 'complete',
      classifier: 'Perceptron',
      step: stepCount,
      w: [...this.w],
      b: this.b,
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

