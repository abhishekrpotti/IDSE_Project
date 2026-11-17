import {describe,it,expect} from 'vitest';
import {freehandBoundaryClass} from '../src/game/boundary.js';

describe('freehand nonlinear boundaries',()=>{
  it('classifies opposite sides of an open curved separator',()=>{
    const curve=[[-1,-.5],[-.5,0],[0,.25],[.5,0],[1,-.5]].map(([x,y])=>({x,y}));
    expect(freehandBoundaryClass([curve],{x:0,y:.8})).toBe(1);
    expect(freehandBoundaryClass([curve],{x:0,y:-.8})).toBe(-1);
  });
  it('keeps a closed stroke as an inside region',()=>{
    const loop=[[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]].map(([x,y])=>({x,y}));
    expect(freehandBoundaryClass([loop],{x:0,y:0})).toBe(1);
    expect(freehandBoundaryClass([loop],{x:1.5,y:0})).toBe(-1);
  });
});
