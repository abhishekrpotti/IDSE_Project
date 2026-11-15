/**
 * Mulberry32 Seedable Pseudo-Random Number Generator
 */

export class RNG {
  constructor(seed = 42) {
    this.s = seed | 0;
  }

  setSeed(seed) {
    this.s = seed | 0;
  }

  /** Returns float in [0, 1) */
  random() {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform float in [min, max) */
  uniform(min, max) {
    return min + this.random() * (max - min);
  }

  /** Standard Gaussian random using Box-Muller */
  gaussian(mean = 0, std = 1) {
    let u1 = this.random();
    let u2 = this.random();
    while (u1 <= 1e-15) u1 = this.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * std;
  }
}

export const globalRNG = new RNG(1337);

