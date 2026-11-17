import {rng} from './rng.js';

export const datasets=['Separable blobs','Overlapping blobs','XOR','Two moons','Circles','Spirals','Correlated blobs','Unequal-variance blobs','Outlier blobs','Checkerboard (3x3)'];
export function makeData(kind='Separable blobs',n=180,seed=12,options={}){
  if(typeof options==='number') options={noise:options};
  const random=rng(seed),out=[],noise=Math.max(0,Math.min(.5,Number(options.noise??.2))),flip=Math.max(0,Math.min(.5,Number(options.labelNoise??options.flipNoise??0)));
  const normal=()=>Math.sqrt(-2*Math.log(Math.max(random(),1e-9)))*Math.cos(2*Math.PI*random());
  for(let i=0;i<n;i++){
    let y=i%2?1:-1,x1,x2,z1=normal(),z2=normal();
    switch(kind){
      case'XOR':{const sx=random()<.5?-1:1,sy=sx*y;x1=sx*.65+z1*(.08+noise*.5);x2=sy*.65+z2*(.08+noise*.5);break}
      case'Two moons':{const t=random()*Math.PI;if(y>0){x1=Math.cos(t);x2=Math.sin(t)}else{x1=1-Math.cos(t);x2=.5-Math.sin(t)}x1+=z1*(noise*.4);x2+=z2*(noise*.4);break}
      case'Circles':{const t=random()*Math.PI*2,rad=y>0?.42:.95;x1=Math.cos(t)*rad+z1*(noise*.25);x2=Math.sin(t)*rad+z2*(noise*.25);break}
      case'Spirals':{const t=random()*Math.PI*3,rad=.12+t*.1,ang=t+(y>0?0:Math.PI);x1=rad*Math.cos(ang)+z1*(noise*.2);x2=rad*Math.sin(ang)+z2*(noise*.2);break}
      case'Correlated blobs':{const a=z1*(.22+noise*.15),b=z1*(.70+noise*.15)+z2*(.16+noise*.08);x1=y*.12+a;x2=b;break}
      case'Unequal-variance blobs':{const spread=y===1?.10:.72;x1=(y===1?-.2:.18)+z1*(spread+noise*.18);x2=(y===1?-.12:.18)+z2*(spread*.82+noise*.16);break}
      case'Outlier blobs':{const outlier=random()<.09;if(outlier){x1=-y*(.75+random()*.55)+z1*.08;x2=z2*.85}else{x1=y*.8+z1*(.13+noise*.18);x2=y*.12+z2*(.17+noise*.18)}break}
      case'Checkerboard (3x3)':{let cx,cy;do{cx=Math.floor(random()*3);cy=Math.floor(random()*3)}while((cx+cy)%2===1&&cx===2&&cy===2);y=(cx+cy)%2===0?1:-1;x1=cx-1+(random()-.5)*.84+z1*noise*.08;x2=cy-1+(random()-.5)*.84+z2*noise*.08;break}
      default:{const sep=kind==='Overlapping blobs'?.3:.8;x1=y*sep+z1*(kind==='Overlapping blobs'?.42+noise*.2:.22+noise*.3);x2=y*.25+z2*(kind==='Overlapping blobs'?.42+noise*.2:.22+noise*.3)}
    }
    if(random()<flip)y=-y;
    out.push({x:[x1,x2],y});
  }
  return out;
}
export function waveData(kind='Separable blobs',n=16,seed=Date.now(),options={}){const r=rng(seed),d=makeData(kind,n,seed,options).map((p,i)=>({...p,id:i,fruit:p.y===1}));for(let i=d.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[d[i],d[j]]=[d[j],d[i]]}return d}

export const datasetInfo={
  'Separable blobs':'Two round-ish clouds sit apart, so a straight boundary is enough. Linear models shine; curved models can still work but add little.',
  'Overlapping blobs':'Two noisy clouds overlap, making some labels ambiguous. Regularized linear models are a good baseline; deep trees and tiny-k neighbors can chase noise.',
  XOR:'Classes occupy alternating diagonal corners. No straight line can separate them; RBF SVM, neighbors, trees, and forests can learn the nonlinear split.',
  'Two moons':'Two crescent-shaped groups interlock. Kernel SVM and k-NN can follow the curved gap; a line underfits and a single tree may need many boxes.',
  Circles:'One class forms a disk and the other a surrounding ring. Radial kernels and neighbors fit this closed boundary; linear models cannot.',
  Spirals:'Two classes wind around the center in alternating arms. Kernel SVM and k-NN can follow local curves; linear models fail and trees need many small regions.',
  'Correlated blobs':'Two tilted clouds share a correlated covariance but have different means. LDA uses the shared full covariance; diagonal Naive Bayes ignores the tilt.',
  'Unequal-variance blobs':'The classes have different spreads. QDA models each covariance separately; LDA and other linear models must compromise with one line.',
  'Outlier blobs':'Mostly separated groups include a few extreme, mislabeled points. Logistic regression and SVM regularization can resist them; Perceptron and k=1 may react strongly.',
  'Checkerboard (3x3)':'Alternating axis-aligned squares make a 3-by-3 checkerboard. Trees and forests match its boxy edges; linear models cannot represent the pattern.'
};
