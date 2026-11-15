import { BaseClassifier } from './base.js';
import { DecisionTree } from './tree.js';
import { RNG } from '../data/rng.js';

export class RandomForest extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.nTrees = options.nTrees ?? 6;
    this.maxDepth = options.maxDepth ?? 3;
    this.trees = [];
    this.rng = new RNG(options.seed ?? 777);
  }

  bootstrapSample(X, y) {
    const n = X.length;
    const sampleX = [];
    const sampleY = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(this.rng.random() * n);
      sampleX.push(X[idx]);
      sampleY.push(y[idx]);
    }
    return { sampleX, sampleY };
  }

  *fitSteps(X, y) {
    this.trees = [];

    for (let t = 0; t < this.nTrees; t++) {
      const { sampleX, sampleY } = this.bootstrapSample(X, y);
      const tree = new DecisionTree({ maxDepth: this.maxDepth });
      tree.fit(sampleX, sampleY);
      this.trees.push(tree);

      yield {
        type: 'step',
        classifier: 'RandomForest',
        step: t + 1,
        treeIndex: t,
        totalTrees: this.nTrees,
        trainedTrees: this.trees.length,
        accuracy: this.calculateAccuracy(X, y)
      };
    }

    this.isFitted = true;

    yield {
      type: 'complete',
      classifier: 'RandomForest',
      step: this.nTrees,
      totalTrees: this.trees.length,
      accuracy: this.calculateAccuracy(X, y)
    };
  }

  predictProba(x) {
    if (this.trees.length === 0) return 0.5;
    let sumProba = 0;
    for (const tree of this.trees) {
      sumProba += tree.predictProba(x);
    }
    return sumProba / this.trees.length;
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

