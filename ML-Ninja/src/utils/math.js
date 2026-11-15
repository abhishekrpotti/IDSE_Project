/**
 * Vector and Matrix Math Utilities for Classical ML
 * Implemented from scratch without external dependencies.
 */

export function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function norm(v) {
  return Math.sqrt(dot(v, v));
}

export function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

export function manhattanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

export function sigmoid(z) {
  // Clamped to avoid overflow
  if (z > 35) return 1;
  if (z < -35) return 0;
  return 1 / (1 + Math.exp(-z));
}

export function mean(X) {
  // X is array of feature vectors
  const n = X.length;
  if (n === 0) return [0, 0];
  const d = X[0].length;
  const mu = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      mu[j] += X[i][j];
    }
  }
  return mu.map(val => val / n);
}

export function covarianceMatrix(X, mu = null) {
  const n = X.length;
  if (n <= 1) return [[1e-4, 0], [0, 1e-4]];
  const d = X[0].length;
  const m = mu || mean(X);
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < d; k++) {
        cov[j][k] += (X[i][j] - m[j]) * (X[i][k] - m[k]);
      }
    }
  }

  // Regularize slightly with ridge epsilon (1e-4) to prevent singular matrices
  for (let j = 0; j < d; j++) {
    for (let k = 0; k < d; k++) {
      cov[j][k] /= (n - 1);
      if (j === k) cov[j][k] += 1e-4;
    }
  }
  return cov;
}

export function invert2x2(M) {
  const a = M[0][0];
  const b = M[0][1];
  const c = M[1][0];
  const d = M[1][1];
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) {
    return [[1, 0], [0, 1]];
  }
  const invDet = 1 / det;
  return [
    [d * invDet, -b * invDet],
    [-c * invDet, a * invDet]
  ];
}

export function det2x2(M) {
  return M[0][0] * M[1][1] - M[0][1] * M[1][0];
}

/**
 * Standardize features: returns { scaledX, means, stds, transform(x) }
 */
export function standardize(X) {
  if (X.length === 0) return { scaledX: [], means: [0, 0], stds: [1, 1], transform: x => x };
  const d = X[0].length;
  const means = mean(X);
  const stds = new Array(d).fill(0);

  for (let i = 0; i < X.length; i++) {
    for (let j = 0; j < d; j++) {
      stds[j] += Math.pow(X[i][j] - means[j], 2);
    }
  }
  for (let j = 0; j < d; j++) {
    stds[j] = Math.sqrt(stds[j] / Math.max(1, X.length - 1)) || 1;
  }

  const transform = (x) => x.map((val, j) => (val - means[j]) / stds[j]);
  const scaledX = X.map(transform);

  return { scaledX, means, stds, transform };
}

/**
 * Given two 2D points (x1, y1) and (x2, y2) from player slash,
 * compute line equation w1 * x + w2 * y + b = 0
 * Normal vector n = (dy, -dx)
 */
export function lineFromSegment(p1, p2) {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const length = Math.sqrt(dx * dx + dy * dy);
  if (length < 1e-6) return { w: [0, 1], b: -p1[1] };

  // Unit normal
  const nx = -dy / length;
  const ny = dx / length;
  // b = -(nx * p1[0] + ny * p1[1])
  const b = -(nx * p1[0] + ny * p1[1]);
  return { w: [nx, ny], b };
}

