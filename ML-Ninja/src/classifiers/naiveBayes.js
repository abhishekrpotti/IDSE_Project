import { BaseClassifier } from './base.js';

export class GaussianNaiveBayes extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.classes = [1, -1];
    this.priors = {};
    this.means = {};
    this.variances = {};
  }

  *fitSteps(X, y) {
    const n = X.length;
    const d = X[0].length;

    // Step 1: Compute priors and separate points by class
    const classPoints = { 1: [], '-1': [] };
    for (let i = 0; i < n; i++) {
      const cls = y[i] >= 0 ? 1 : -1;
      classPoints[cls].push(X[i]);
    }

    this.priors[1] = classPoints[1].length / n;
    this.priors[-1] = classPoints[-1].length / n;

    yield {
      type: 'step',
      classifier: 'GaussianNaiveBayes',
      step: 1,
      phase: 'Priors Computed',
      priors: { ...this.priors },
      accuracy: 0.5
    };

    // Step 2: Compute means for each class
    for (const cls of this.classes) {
      const pts = classPoints[cls];
      const mu = new Array(d).fill(0);
      for (const pt of pts) {
        for (let j = 0; j < d; j++) {
          mu[j] += pt[j];
        }
      }
      this.means[cls] = mu.map(val => val / Math.max(1, pts.length));
    }

    yield {
      type: 'step',
      classifier: 'GaussianNaiveBayes',
      step: 2,
      phase: 'Means Computed',
      means: { 1: [...this.means[1]], '-1': [...this.means[-1]] },
      accuracy: 0.5
    };

    // Step 3: Compute variances
    for (const cls of this.classes) {
      const pts = classPoints[cls];
      const mu = this.means[cls];
      const variance = new Array(d).fill(0);
      for (const pt of pts) {
        for (let j = 0; j < d; j++) {
          variance[j] += Math.pow(pt[j] - mu[j], 2);
        }
      }
      // Add small epsilon to prevent 0 variance
      this.variances[cls] = variance.map(v => v / Math.max(1, pts.length - 1) + 1e-3);
    }

    this.isFitted = true;

    yield {
      type: 'complete',
      classifier: 'GaussianNaiveBayes',
      step: 3,
      phase: 'Variances & Ellipses Computed',
      means: this.means,
      variances: this.variances,
      priors: this.priors,
      accuracy: this.calculateAccuracy(X, y)
    };
  }

  gaussianPdf(x, mean, variance) {
    const exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
    return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
  }

  logPosterior(x, cls) {
    const d = x.length;
    let logProb = Math.log(Math.max(1e-12, this.priors[cls]));
    for (let j = 0; j < d; j++) {
      const pdf = this.gaussianPdf(x[j], this.means[cls][j], this.variances[cls][j]);
      logProb += Math.log(Math.max(1e-12, pdf));
    }
    return logProb;
  }

  predictProba(x) {
    const logP1 = this.logPosterior(x, 1);
    const logPMinus1 = this.logPosterior(x, -1);
    // Softmax over 2 log-probabilities
    const maxLog = Math.max(logP1, logPMinus1);
    const exp1 = Math.exp(logP1 - maxLog);
    const exp2 = Math.exp(logPMinus1 - maxLog);
    return exp1 / (exp1 + exp2);
  }

  predict(x) {
    return this.predictProba(x) >= 0.5 ? 1 : -1;
  }

  calculateAccuracy(X, y) {
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      if (this.predict(X[i]) === y[i]) correct++;
    }
    return correct / (X.length || 1);
  }
}

