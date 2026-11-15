import { BaseClassifier } from './base.js';
import { mean, covarianceMatrix, invert2x2, det2x2, dot } from '../utils/math.js';

export class DiscriminantAnalysis extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.mode = options.mode ?? 'lda'; // 'lda' | 'qda'
    this.classes = [1, -1];
    this.means = {};
    this.covariances = {};
    this.invCovariances = {};
    this.priors = {};
    this.pooledCov = null;
    this.invPooledCov = null;
  }

  *fitSteps(X, y) {
    const n = X.length;
    const pts1 = [];
    const pts2 = [];
    for (let i = 0; i < n; i++) {
      if (y[i] >= 0) pts1.push(X[i]);
      else pts2.push(X[i]);
    }

    this.priors[1] = pts1.length / n;
    this.priors[-1] = pts2.length / n;
    this.means[1] = mean(pts1);
    this.means[-1] = mean(pts2);

    yield {
      type: 'step',
      classifier: this.mode === 'lda' ? 'LDA' : 'QDA',
      step: 1,
      phase: 'Computed Class Means',
      means: this.means,
      accuracy: 0.5
    };

    const cov1 = covarianceMatrix(pts1, this.means[1]);
    const cov2 = covarianceMatrix(pts2, this.means[-1]);
    this.covariances[1] = cov1;
    this.covariances[-1] = cov2;
    this.invCovariances[1] = invert2x2(cov1);
    this.invCovariances[-1] = invert2x2(cov2);

    // Pooled covariance for LDA
    const d = X[0].length;
    this.pooledCov = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let r = 0; r < d; r++) {
      for (let c = 0; c < d; c++) {
        this.pooledCov[r][c] =
          ((pts1.length - 1) * cov1[r][c] + (pts2.length - 1) * cov2[r][c]) /
          Math.max(1, n - 2);
      }
    }
    this.invPooledCov = invert2x2(this.pooledCov);

    this.isFitted = true;

    yield {
      type: 'complete',
      classifier: this.mode === 'lda' ? 'LDA' : 'QDA',
      step: 2,
      phase: 'Covariance Matrices & Boundary Determined',
      means: this.means,
      covariances: this.covariances,
      pooledCov: this.pooledCov,
      accuracy: this.calculateAccuracy(X, y)
    };
  }

  discriminantScore(x, cls) {
    const mu = this.means[cls];
    const prior = Math.max(1e-12, this.priors[cls]);

    if (this.mode === 'lda') {
      // Linear: x^T \Sigma^-1 \mu - 0.5 \mu^T \Sigma^-1 \mu + log(prior)
      const inv = this.invPooledCov;
      const invMu = [
        inv[0][0] * mu[0] + inv[0][1] * mu[1],
        inv[1][0] * mu[0] + inv[1][1] * mu[1]
      ];
      const term1 = dot(x, invMu);
      const term2 = 0.5 * dot(mu, invMu);
      return term1 - term2 + Math.log(prior);
    } else {
      // QDA: -0.5 * log|cov| - 0.5 * (x - mu)^T \Sigma^-1 (x - mu) + log(prior)
      const cov = this.covariances[cls];
      const inv = this.invCovariances[cls];
      const diff = [x[0] - mu[0], x[1] - mu[1]];
      const invDiff = [
        inv[0][0] * diff[0] + inv[0][1] * diff[1],
        inv[1][0] * diff[0] + inv[1][1] * diff[1]
      ];
      const mahalanobis = dot(diff, invDiff);
      const det = Math.max(1e-12, det2x2(cov));
      return -0.5 * Math.log(det) - 0.5 * mahalanobis + Math.log(prior);
    }
  }

  predictProba(x) {
    const d1 = this.discriminantScore(x, 1);
    const d2 = this.discriminantScore(x, -1);
    const maxD = Math.max(d1, d2);
    const exp1 = Math.exp(d1 - maxD);
    const exp2 = Math.exp(d2 - maxD);
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

