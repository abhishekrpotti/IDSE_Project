import {describe,it,expect} from 'vitest';
import {createRoundStats,recordRoundHit,finalizeRound} from '../src/game/roundStats.js';
import {linearBoundaryClass} from '../src/game/boundary.js';

const modelFrom=predictions=>({predict:([index])=>predictions[index]});
const drawnLine={type:'line',points:[{x:0,y:-1},{x:0,y:1}]};

describe('round result stats',()=>{
  it('computes model accuracy from the same predictions shown in the confusion matrix',()=>{
    const labels=[1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1];
    const predictions=[1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
    const points=labels.map((y,index)=>({x:[index],y}));
    const stats=finalizeRound(createRoundStats(),{points,model:modelFrom(predictions),playerPredict:()=>-1,userBoundary:null});
    expect(stats.confusionMatrix).toEqual({tn:8,fp:0,fn:1,tp:7});
    expect(stats.modelAccuracy).toBe(15/16);
  });

  it('turns a signed line-side value into a class label',()=>{
    const a={x:0,y:-1},b={x:0,y:1};
    expect(linearBoundaryClass(a,b,{x:-.5,y:0})).toBe(1);
    expect(linearBoundaryClass(a,b,{x:.5,y:0})).toBe(-1);
  });

  it('computes user accuracy from a known boundary and point set',()=>{
    const points=[{x:[0],y:1},{x:[1],y:1},{x:[2],y:-1},{x:[3],y:-1}];
    const stats=createRoundStats();recordRoundHit(stats,1);recordRoundHit(stats,1);
    finalizeRound(stats,{points,model:modelFrom([1,1,-1,-1]),playerPredict:([x],flip)=>flip?(x<2?-1:1):(x<2?1:-1),userBoundary:drawnLine});
    expect(stats.userAccuracy).toBe(1);
    expect(stats.sliced).toBe(2);expect(stats.bombsHit).toBe(0);expect(stats.modelAgreement).toBe(1);expect(stats.agreementScore).toBe(100);expect(stats.score).toBe(120);
    expect(stats.points).toHaveLength(4);expect(stats.userBoundary).toEqual(drawnLine);
    expect(stats.modelPredictionGrid).toHaveLength(64);expect(stats.modelPredictionGrid[0]).toHaveLength(64);
  });

  it('reports zero accuracy when no boundary was drawn',()=>{
    const points=[{x:[0],y:1},{x:[1],y:-1}];
    const stats=finalizeRound(createRoundStats(),{points,model:modelFrom([1,-1]),playerPredict:()=>-1,userBoundary:null});
    expect(stats.userAccuracy).toBe(0);expect(stats.boundaryDrawn).toBe(false);expect(stats.verdict).toBe('No boundary drawn');
  });

  it('increments hit counters and resets them for a new round',()=>{
    const current=createRoundStats();recordRoundHit(current,1);recordRoundHit(current,-1);
    expect(current).toMatchObject({sliced:1,bombsHit:1,score:2});
    const next=createRoundStats();expect(next).toMatchObject({sliced:0,bombsHit:0,score:0,userAccuracy:0,modelAccuracy:0,userBoundary:null,points:[]});
  });

  it('changes round score with the number and type of hit events',()=>{
    const oneFruit=createRoundStats();recordRoundHit(oneFruit,1);
    const twoFruitOneBomb=createRoundStats();recordRoundHit(twoFruitOneBomb,1);recordRoundHit(twoFruitOneBomb,1);recordRoundHit(twoFruitOneBomb,-1);
    expect(oneFruit.score).toBe(10);expect(twoFruitOneBomb.score).toBe(12);expect(oneFruit.score).not.toBe(twoFruitOneBomb.score);
  });

  it('produces distinct results for two rounds with different inputs',()=>{
    const first=createRoundStats();recordRoundHit(first,1);recordRoundHit(first,1);
    finalizeRound(first,{points:[{x:[0],y:1},{x:[1],y:-1}],model:modelFrom([1,-1]),playerPredict:([x])=>x===0?1:-1,userBoundary:drawnLine});
    const second=finalizeRound(createRoundStats(),{points:[{x:[0],y:1},{x:[1],y:-1}],model:modelFrom([-1,1]),playerPredict:()=>-1,userBoundary:null});
    expect(first).not.toBe(second);expect(first.userAccuracy).not.toBe(second.userAccuracy);expect(first.modelAccuracy).not.toBe(second.modelAccuracy);expect(first.score).not.toBe(second.score);
  });
});
