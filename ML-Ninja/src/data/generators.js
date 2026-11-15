/**
 * 2D & 3D Synthetic Dataset Generators for ML Ninja
 */
import { RNG } from './rng.js';

export function generateDataset(type = 'separable', options = {}) {
  const {
    n = 60,
    noise = 0.15,
    seed = 42,
    dimensions = 2,
    scale = 3.5 // Scale to fit into [-4, 4] world coordinates
  } = options;

  const rng = new RNG(seed);
  const X = [];
  const y = [];

  switch (type) {
    case 'separable': {
      // Two well-separated Gaussian blobs
      const nPerClass = Math.floor(n / 2);
      // Class +1: center (-1.8, -1.2)
      for (let i = 0; i < nPerClass; i++) {
        const pt = [
          -1.8 + rng.gaussian(0, noise * 1.5),
          -1.2 + rng.gaussian(0, noise * 1.5)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise));
        X.push(pt);
        y.push(1);
      }
      // Class -1: center (1.8, 1.2)
      for (let i = 0; i < n - nPerClass; i++) {
        const pt = [
          1.8 + rng.gaussian(0, noise * 1.5),
          1.2 + rng.gaussian(0, noise * 1.5)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise));
        X.push(pt);
        y.push(-1);
      }
      break;
    }

    case 'overlapping': {
      const nPerClass = Math.floor(n / 2);
      for (let i = 0; i < nPerClass; i++) {
        const pt = [
          -0.8 + rng.gaussian(0, 0.8 + noise),
          -0.6 + rng.gaussian(0, 0.8 + noise)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, 0.5));
        X.push(pt);
        y.push(1);
      }
      for (let i = 0; i < n - nPerClass; i++) {
        const pt = [
          0.8 + rng.gaussian(0, 0.8 + noise),
          0.6 + rng.gaussian(0, 0.8 + noise)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, 0.5));
        X.push(pt);
        y.push(-1);
      }
      break;
    }

    case 'xor': {
      // 4 quadrants: Q1 (+1), Q2 (-1), Q3 (+1), Q4 (-1)
      const nPerQuadrant = Math.floor(n / 4);
      const centers = [
        { c: [1.8, 1.8], label: 1 },
        { c: [-1.8, 1.8], label: -1 },
        { c: [-1.8, -1.8], label: 1 },
        { c: [1.8, -1.8], label: -1 }
      ];
      for (const quad of centers) {
        for (let i = 0; i < nPerQuadrant; i++) {
          const pt = [
            quad.c[0] + rng.gaussian(0, 0.45 + noise),
            quad.c[1] + rng.gaussian(0, 0.45 + noise)
          ];
          if (dimensions === 3) pt.push(rng.gaussian(0, 0.4));
          X.push(pt);
          y.push(quad.label);
        }
      }
      break;
    }

    case 'moons': {
      const nPerClass = Math.floor(n / 2);
      // Top moon: class +1
      for (let i = 0; i < nPerClass; i++) {
        const theta = (i / nPerClass) * Math.PI;
        const pt = [
          2.0 * Math.cos(theta) - 1.0 + rng.gaussian(0, noise * 0.8),
          2.0 * Math.sin(theta) - 0.5 + rng.gaussian(0, noise * 0.8)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise * 0.5));
        X.push(pt);
        y.push(1);
      }
      // Bottom moon: class -1
      for (let i = 0; i < n - nPerClass; i++) {
        const theta = (i / (n - nPerClass)) * Math.PI;
        const pt = [
          2.0 * Math.cos(theta) + 1.0 + rng.gaussian(0, noise * 0.8),
          -2.0 * Math.sin(theta) + 0.5 + rng.gaussian(0, noise * 0.8)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise * 0.5));
        X.push(pt);
        y.push(-1);
      }
      break;
    }

    case 'circles': {
      const nPerClass = Math.floor(n / 2);
      // Inner circle: class +1 (radius ~1.2)
      for (let i = 0; i < nPerClass; i++) {
        const r = 1.1 + rng.gaussian(0, noise * 0.6);
        const theta = rng.uniform(0, Math.PI * 2);
        const pt = [r * Math.cos(theta), r * Math.sin(theta)];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise * 0.5));
        X.push(pt);
        y.push(1);
      }
      // Outer ring: class -1 (radius ~2.8)
      for (let i = 0; i < n - nPerClass; i++) {
        const r = 2.7 + rng.gaussian(0, noise * 0.6);
        const theta = rng.uniform(0, Math.PI * 2);
        const pt = [r * Math.cos(theta), r * Math.sin(theta)];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise * 0.5));
        X.push(pt);
        y.push(-1);
      }
      break;
    }

    case 'spirals': {
      const nPerClass = Math.floor(n / 2);
      // Spiral 1 (+1)
      for (let i = 0; i < nPerClass; i++) {
        const theta = (i / nPerClass) * 2.5 * Math.PI;
        const r = (theta / (2.5 * Math.PI)) * 3.2;
        const pt = [
          r * Math.cos(theta) + rng.gaussian(0, noise * 0.5),
          r * Math.sin(theta) + rng.gaussian(0, noise * 0.5)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise * 0.4));
        X.push(pt);
        y.push(1);
      }
      // Spiral 2 (-1) offset by PI
      for (let i = 0; i < n - nPerClass; i++) {
        const theta = (i / (n - nPerClass)) * 2.5 * Math.PI;
        const r = (theta / (2.5 * Math.PI)) * 3.2;
        const pt = [
          r * Math.cos(theta + Math.PI) + rng.gaussian(0, noise * 0.5),
          r * Math.sin(theta + Math.PI) + rng.gaussian(0, noise * 0.5)
        ];
        if (dimensions === 3) pt.push(rng.gaussian(0, noise * 0.4));
        X.push(pt);
        y.push(-1);
      }
      break;
    }

    default:
      throw new Error(`Unknown dataset type: ${type}`);
  }

  // Shuffle dataset reproducibly
  const indices = Array.from({ length: X.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  return {
    X: indices.map(i => X[i]),
    y: indices.map(i => y[i]),
    type,
    options
  };
}

