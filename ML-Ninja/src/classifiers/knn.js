import { BaseClassifier } from './base.js';
import { euclideanDistance, manhattanDistance } from '../utils/math.js';

export class KNN extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.k = options.k ?? 3;
    this.metric = options.metric ?? 'euclidean'; // 'euclidean' | 'manhattan'
    this.X = [];
    this.y = [];
  }

  getDist(a, b) {
    return this.metric === 'manhattan' ? manhattanDistance(a, b) : euclideanDistance(a, b);
  }

  getNeighbors(x) {
    if (this.X.length === 0) return [];
    const distances = this.X.map((pt, i) => ({
      index: i,
      point: pt,
      label: this.y[i],
      dist: this.getDist(x, pt)
    }));

    distances.sort((a, b) => a.dist - b.dist);
    return distances.slice(0, Math.min(this.k, distances.length));
  }

  predictProba(x) {
    const neighbors = this.getNeighbors(x);
    if (neighbors.length === 0) return 0.5;
    let posCount = 0;
    for (const n of neighbors) {
      if (n.label >= 0) posCount++;
    }
    return posCount / neighbors.length;
  }

  predict(x) {
    return this.predictProba(x) >= 0.5 ? 1 : -1;
  }

  *fitSteps(X, y) {
    this.X = [];
    this.y = [];
    // Animate storing points into instance memory
    for (let i = 0; i < X.length; i++) {
      this.X.push(X[i]);
      this.y.push(y[i]);

      if (i % 3 === 0 || i === X.length - 1) {
        yield {
          type: 'step',
          classifier: 'KNN',
          step: i + 1,
          storedPoints: this.X.length,
          totalPoints: X.length,
          k: this.k,
          metric: this.metric,
          accuracy: this.calculateAccuracy(X, y)
        };
      }
    }

    this.isFitted = true;
    yield {
      type: 'complete',
      classifier: 'KNN',
      step: X.length,
      storedPoints: this.X.length,
      k: this.k,
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

