import {confusion} from '../utils/metrics.js';

const GRID_SIZE=64;

export function createRoundStats() {
  return {
    sliced: 0,
    bombsHit: 0,
    score: 0,
    userBoundary: null,
    points: [],
    userAccuracy: 0,
    modelAccuracy: 0,
    modelAgreement: 0,
    agreementScore: 0,
    confusionMatrix: {tn:0,fp:0,fn:0,tp:0},
    modelPredictionGrid: [],
    total: 0,
    verdict: 'Missed the mark',
    flipped: false,
    boundaryDrawn: false,
    finalized: false,
  };
}

export function recordRoundHit(stats,label) {
  if(stats.finalized) throw new Error('Cannot record a hit after the round is finalized');
  if(label===1){stats.sliced+=1;stats.score+=10}
  else if(label===-1){stats.bombsHit+=1;stats.score-=8}
  else throw new TypeError(`Unknown point label: ${label}`);
  return stats;
}

function hasBoundary(boundary) {
  if(!boundary)return false;
  if(boundary.type==='line'){
    const [a,b]=boundary.points??[];
    return Boolean(a&&b&&Math.hypot(b.x-a.x,b.y-a.y)>.05);
  }
  return (boundary.strokes??[]).some(stroke=>{
    if(stroke.length<2)return false;
    let length=0;for(let i=1;i<stroke.length;i++)length+=Math.hypot(stroke[i].x-stroke[i-1].x,stroke[i].y-stroke[i-1].y);
    return length>.05;
  });
}

/** Finalize one wave once, preserving hit counters recorded by real swipe events. */
export function finalizeRound(stats,{points,model,playerPredict,userBoundary}) {
  if(stats.finalized)return stats;
  const wavePoints=(points??[]).map(point=>({id:point.id,x:[...point.x],y:point.y}));
  const truth=wavePoints.map(point=>point.y);
  const modelPredictions=wavePoints.map(point=>model.predict(point.x));
  const matrix=confusion(truth,modelPredictions);
  const confusionMatrix={tp:matrix[0][0],fn:matrix[0][1],fp:matrix[1][0],tn:matrix[1][1]};
  const total=truth.length;
  const matrixTotal=confusionMatrix.tp+confusionMatrix.tn+confusionMatrix.fp+confusionMatrix.fn;
  const modelAccuracy=matrixTotal===0?0:(confusionMatrix.tp+confusionMatrix.tn)/matrixTotal;
  const boundary=userBoundary?structuredClone(userBoundary):null;
  const boundaryDrawn=hasBoundary(boundary);
  let best={accuracy:0,flipped:false,predictions:[]};
  if(boundaryDrawn){
    for(const flipped of [false,true]){
      const predictions=wavePoints.map(point=>playerPredict(point.x,flipped));
      const correct=truth.reduce((count,label,index)=>count+Number(predictions[index]===label),0);
      const userAccuracy=total===0?0:correct/total;
      if(userAccuracy>best.accuracy)best={accuracy:userAccuracy,flipped,predictions};
    }
  } else {
    best={accuracy:0,flipped:false,predictions:wavePoints.map(point=>playerPredict(point.x,false))};
  }
  const modelPredictionGrid=Array.from({length:GRID_SIZE},(_,row)=>Array.from({length:GRID_SIZE},(_,col)=>{
    const x=(col/(GRID_SIZE-1))*4-2;
    const y=2-(row/(GRID_SIZE-1))*4;
    return model.predict([x,y]);
  }));
  const agreementCount=boundaryDrawn?best.predictions.reduce((count,prediction,index)=>count+Number(prediction===modelPredictions[index]),0):0;
  const modelAgreement=total===0?0:agreementCount/total;
  const agreementScore=Math.round(modelAgreement*100);
  const modelScoreGrid=model.metadata?.name==='Kernel SVM'&&typeof model.score==='function'?Array.from({length:GRID_SIZE},(_,row)=>Array.from({length:GRID_SIZE},(_,col)=>{const x=(col/(GRID_SIZE-1))*4-2,y=2-(row/(GRID_SIZE-1))*4;return model.score([x,y])})):[];
  const modelProbabilityGrid=typeof model.predictProba==='function'?Array.from({length:GRID_SIZE},(_,row)=>Array.from({length:GRID_SIZE},(_,col)=>{
    const x=(col/(GRID_SIZE-1))*4-2;
    const y=2-(row/(GRID_SIZE-1))*4;
    return model.predictProba([x,y]);
  })):[];
  Object.assign(stats,{
    userBoundary:boundary,
    points:wavePoints,
    userAccuracy:best.accuracy,
    modelAccuracy,
    modelPredictions,
    modelAgreement,
    agreementScore,
    score:stats.score+agreementScore,
    confusionMatrix,
    modelPredictionGrid,
    modelProbabilityGrid,
    modelScoreGrid,
    modelSupportVectors:(model.supportVectors??[]).map(vector=>({x:[...vector.x],y:vector.y})),
    modelLinearSvm:model.metadata?.name==='Linear SVM'&&model.w&&model.scaler?{w:[...model.w],b:model.b,mean:[...model.scaler.mean],scale:[...model.scaler.scale]}:null,
    total,
    flipped:best.flipped,
    boundaryDrawn,
    verdict:!boundaryDrawn?'No boundary drawn':best.accuracy>=.9||best.accuracy>=modelAccuracy-.05?'Nice cut!':best.accuracy>=.75?'Almost there!':'Missed the mark',
    finalized:true,
  });
  return stats;
}
