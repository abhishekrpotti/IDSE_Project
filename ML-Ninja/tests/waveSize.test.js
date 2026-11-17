import {describe,it,expect} from 'vitest';
import {waveData} from '../src/data/generators.js';

describe('wave point count',()=>{
  it.each([16,24,32])('creates a balanced %i-point wave',size=>{
    const points=waveData('Separable blobs',size,123);
    expect(points).toHaveLength(size);
    expect(points.filter(point=>point.y===1)).toHaveLength(size/2);
    expect(points.filter(point=>point.y===-1)).toHaveLength(size/2);
  });
});
