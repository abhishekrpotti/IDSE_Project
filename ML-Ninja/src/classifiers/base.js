/**
 * Base Classifier Interface
 */

export class BaseClassifier {
  constructor(options = {}) {
    this.options = options;
    this.isFitted = false;
  }

  fit(X, y) {
    const generator = this.fitSteps(X, y);
    let result = generator.next();
    while (!result.done) {
      result = generator.next();
    }
    this.isFitted = true;
    return this;
  }

  predict(x) {
    throw new Error('predict(x) must be implemented by subclass');
  }

  predictProba(x) {
    // Default fallback: hard probability 0 or 1
    const p = this.predict(x);
    return p >= 0 ? 1 : 0;
  }

  *fitSteps(X, y) {
    throw new Error('*fitSteps(X, y) must be implemented by subclass');
  }
}

