import { BaseClassifier } from './base.js';

export class DecisionTree extends BaseClassifier {
  constructor(options = {}) {
    super(options);
    this.maxDepth = options.maxDepth ?? 4;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.criterion = options.criterion ?? 'gini'; // 'gini' | 'entropy'
    this.root = null;
    this.splits = []; // List of all split boundaries { feature, threshold, bounds, depth }
  }

  impurity(labels) {
    if (labels.length === 0) return 0;
    let pos = 0;
    for (const val of labels) {
      if (val >= 0) pos++;
    }
    const p1 = pos / labels.length;
    const p2 = 1 - p1;

    if (this.criterion === 'entropy') {
      const e1 = p1 > 1e-9 ? -p1 * Math.log2(p1) : 0;
      const e2 = p2 > 1e-9 ? -p2 * Math.log2(p2) : 0;
      return e1 + e2;
    }
    // Gini
    return 1 - (p1 * p1 + p2 * p2);
  }

  findBestSplit(indices, X, y) {
    let bestGain = -1;
    let bestSplit = null;
    const n = indices.length;
    const currentImpurity = this.impurity(indices.map(i => y[i]));

    const d = X[0].length;
    for (let feature = 0; feature < d; feature++) {
      // Find candidate thresholds
      const values = indices.map(i => X[i][feature]);
      values.sort((a, b) => a - b);

      for (let k = 0; k < values.length - 1; k++) {
        if (Math.abs(values[k] - values[k + 1]) < 1e-6) continue;
        const threshold = (values[k] + values[k + 1]) / 2;

        const left = [];
        const right = [];
        for (const idx of indices) {
          if (X[idx][feature] <= threshold) left.push(idx);
          else right.push(idx);
        }

        if (left.length === 0 || right.length === 0) continue;

        const impLeft = this.impurity(left.map(i => y[i]));
        const impRight = this.impurity(right.map(i => y[i]));
        const gain = currentImpurity - ((left.length / n) * impLeft + (right.length / n) * impRight);

        if (gain > bestGain) {
          bestGain = gain;
          bestSplit = { feature, threshold, left, right, gain };
        }
      }
    }

    return bestSplit;
  }

  *fitSteps(X, y) {
    this.splits = [];
    const rootIndices = Array.from({ length: X.length }, (_, i) => i);

    // Queue for breadth-first construction with step-by-step yield
    const queue = [{
      nodeParent: null,
      isLeft: false,
      indices: rootIndices,
      depth: 0,
      bounds: { xMin: -5, xMax: 5, yMin: -5, yMax: 5 }
    }];

    this.root = { isLeaf: true, proba: this.computeProba(rootIndices, y) };
    let stepCount = 0;

    while (queue.length > 0) {
      const current = queue.shift();
      const { indices, depth, bounds } = current;
      const proba = this.computeProba(indices, y);

      if (depth >= this.maxDepth || indices.length < this.minSamplesSplit || proba === 0 || proba === 1) {
        const leafNode = { isLeaf: true, proba, samples: indices.length };
        if (current.nodeParent) {
          if (current.isLeft) current.nodeParent.left = leafNode;
          else current.nodeParent.right = leafNode;
        } else {
          this.root = leafNode;
        }
        continue;
      }

      const split = this.findBestSplit(indices, X, y);
      if (!split || split.gain <= 1e-5) {
        const leafNode = { isLeaf: true, proba, samples: indices.length };
        if (current.nodeParent) {
          if (current.isLeft) current.nodeParent.left = leafNode;
          else current.nodeParent.right = leafNode;
        } else {
          this.root = leafNode;
        }
        continue;
      }

      // Create decision split node
      const node = {
        isLeaf: false,
        feature: split.feature,
        threshold: split.threshold,
        left: null,
        right: null,
        depth
      };

      if (current.nodeParent) {
        if (current.isLeft) current.nodeParent.left = node;
        else current.nodeParent.right = node;
      } else {
        this.root = node;
      }

      stepCount++;
      this.splits.push({
        feature: split.feature,
        threshold: split.threshold,
        depth,
        bounds: { ...bounds }
      });

      yield {
        type: 'step',
        classifier: 'DecisionTree',
        step: stepCount,
        split: {
          feature: split.feature,
          threshold: split.threshold,
          depth,
          gain: split.gain
        },
        splits: [...this.splits],
        accuracy: this.calculateAccuracy(X, y)
      };

      // Children bounds
      const leftBounds = { ...bounds };
      const rightBounds = { ...bounds };
      if (split.feature === 0) {
        leftBounds.xMax = split.threshold;
        rightBounds.xMin = split.threshold;
      } else {
        leftBounds.yMax = split.threshold;
        rightBounds.yMin = split.threshold;
      }

      queue.push({
        nodeParent: node,
        isLeft: true,
        indices: split.left,
        depth: depth + 1,
        bounds: leftBounds
      });

      queue.push({
        nodeParent: node,
        isLeft: false,
        indices: split.right,
        depth: depth + 1,
        bounds: rightBounds
      });
    }

    this.isFitted = true;

    yield {
      type: 'complete',
      classifier: 'DecisionTree',
      step: stepCount,
      splits: [...this.splits],
      accuracy: this.calculateAccuracy(X, y)
    };
  }

  computeProba(indices, y) {
    if (indices.length === 0) return 0.5;
    let pos = 0;
    for (const idx of indices) {
      if (y[idx] >= 0) pos++;
    }
    return pos / indices.length;
  }

  predictProba(x) {
    if (!this.root) return 0.5;
    let node = this.root;
    while (!node.isLeaf) {
      if (x[node.feature] <= node.threshold) {
        node = node.left || node;
      } else {
        node = node.right || node;
      }
    }
    return node.proba ?? 0.5;
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

