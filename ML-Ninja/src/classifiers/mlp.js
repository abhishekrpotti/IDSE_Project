import { BaseClassifier } from './base.js';
import { dot, sigmoid } from '../utils/math.js';
import { RNG } from '../data/rng.js';

export class MLP extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.hiddenDim = options.hiddenDim ?? 8;
    this.lr = options.lr ?? 0.15;
    this.epochs = options.epochs ?? 80;
    this.rng = new RNG(options.seed ?? 101);

    // Weights: W1: (inputDim x hiddenDim), b1: (hiddenDim), W2: (hiddenDim x 1), b2: 1
    this.W1 = [];
    this.b1 = [];
    this.W2 = [];
    this.b2 = 0;
    this.lossHistory = [];
  }

  initWeights(inputDim) {
    const scale = Math.sqrt(2 / (inputDim + this.hiddenDim));
    this.W1 = Array.from({ length: inputDim }, () =>
      Array.from({ length: this.hiddenDim }, () => this.rng.gaussian(0, scale))
    );
    this.b1 = new Array(this.hiddenDim).fill(0);
    this.W2 = Array.from({ length: this.hiddenDim }, () => this.rng.gaussian(0, 0.5));
    this.b2 = 0;
  }

  forward(x) {
    // Hidden layer with Tanh
    const hidden = new Array(this.hiddenDim).fill(0);
    for (let h = 0; h < this.hiddenDim; h++) {
      let sum = this.b1[h];
      for (let i = 0; i < x.length; i++) {
        sum += x[i] * this.W1[i][h];
      }
      hidden[h] = Math.tanh(sum);
    }

    // Output with Sigmoid
    let outZ = this.b2;
    for (let h = 0; h < this.hiddenDim; h++) {
      outZ += hidden[h] * this.W2[h];
    }
    const output = sigmoid(outZ);

    return { hidden, output };
  }

  predictProba(x) {
    if (this.W1.length === 0) return 0.5;
    return this.forward(x).output;
  }

  predict(x) {
    return this.predictProba(x) >= 0.5 ? 1 : -1;
  }

  *fitSteps(X, y) {
    const n = X.length;
    const inputDim = X[0].length;
    this.initWeights(inputDim);
    this.lossHistory = [];

    // Target values in {0, 1}
    const t = y.map(val => (val >= 0 ? 1 : 0));

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < n; i++) {
        const xi = X[i];
        const ti = t[i];
        const { hidden, output } = this.forward(xi);

        // Binary Cross Entropy Loss
        const pClamped = Math.max(1e-12, Math.min(1 - 1e-12, output));
        totalLoss += -(ti * Math.log(pClamped) + (1 - ti) * Math.log(1 - pClamped));

        // Backprop
        // dL/d(outZ) = output - ti (for sigmoid + BCE)
        const dOutZ = output - ti;

        // Gradient for W2 and b2
        const dW2 = hidden.map(h => dOutZ * h);
        const db2 = dOutZ;

        // Gradient for hidden layer
        const dHidden = new Array(this.hiddenDim).fill(0);
        for (let h = 0; h < this.hiddenDim; h++) {
          dHidden[h] = dOutZ * this.W2[h] * (1 - hidden[h] * hidden[h]); // dtanh
        }

        // Apply SGD updates
        for (let h = 0; h < this.hiddenDim; h++) {
          this.W2[h] -= this.lr * dW2[h];
          this.b2 -= this.lr * db2 * 0.5;

          for (let j = 0; j < inputDim; j++) {
            this.W1[j][h] -= this.lr * dHidden[h] * xi[j];
          }
          this.b1[h] -= this.lr * dHidden[h] * 0.5;
        }
      }

      const loss = totalLoss / n;
      this.lossHistory.push(loss);

      if (epoch % 2 === 0 || epoch === this.epochs - 1) {
        yield {
          type: 'step',
          classifier: 'MLP',
          step: epoch + 1,
          epoch: epoch + 1,
          loss,
          lossHistory: [...this.lossHistory],
          accuracy: this.calculateAccuracy(X, y)
        };
      }
    }

    this.isFitted = true;

    yield {
      type: 'complete',
      classifier: 'MLP',
      step: this.epochs,
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

