import { calculateMetrics } from '../utils/metrics.js';

export class ScoringSystem {
  constructor() {
    this.score = 0;
    this.lives = 3;
    this.combo = 0;
    this.comboTimer = 0;
    this.playerBoundaryAccuracy = null;
    this.modelAccuracy = null;
  }

  reset() {
    this.score = 0;
    this.lives = 3;
    this.combo = 0;
    this.comboTimer = 0;
    this.playerBoundaryAccuracy = null;
  }

  onFruitSliced() {
    this.combo++;
    this.comboTimer = 0.8;
    const points = 10 * Math.max(1, this.combo);
    this.score += points;
    return { points, combo: this.combo };
  }

  onBombHit() {
    this.lives = Math.max(0, this.lives - 1);
    this.combo = 0;
    return { lives: this.lives, isGameOver: this.lives <= 0 };
  }

  update(dt) {
    if (this.comboTimer > 0) {
      this.comboTimer -= dt;
      if (this.comboTimer <= 0) {
        this.combo = 0;
      }
    }
  }

  /**
   * Evaluates player's hand-drawn slash boundary line w1*x + w2*y + b = 0
   * against active points or dataset points.
   */
  evaluatePlayerLine(w, b, points) {
    if (!points || points.length === 0) return 0;
    let correct = 0;
    for (const pt of points) {
      const x = pt.userData ? [pt.position.x, pt.position.y] : pt[0];
      const yTrue = pt.userData ? pt.userData.label : pt[1];

      const score = w[0] * x[0] + w[1] * x[1] + b;
      const pred = score >= 0 ? 1 : -1;
      if (pred === yTrue) correct++;
    }

    const acc = correct / points.length;
    // Check if inverted gives better score
    const invertedAcc = 1.0 - acc;
    this.playerBoundaryAccuracy = Math.max(acc, invertedAcc);
    return this.playerBoundaryAccuracy;
  }
}
