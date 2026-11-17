import {datasets,makeData} from '../src/data/generators.js';
import {classifierNames,trainClassifier} from '../src/classifiers/index.js';

const seeds=Number(process.env.BENCH_SEEDS??20),trainSize=120,testSize=180;
const rows=[];
for(const modelName of classifierNames){
  const cells=[];
  for(const dataset of datasets){
    const scores=[];
    for(let seed=1;seed<=seeds;seed++){
      const training=makeData(dataset,trainSize,seed,{noise:.2});
      const test=makeData(dataset,testSize,seed+10000,{noise:.2});
      const model=trainClassifier(modelName,training,{},seed);
      scores.push(test.reduce((n,p)=>n+Number(model.predict(p.x)===p.y),0)/test.length);
    }
    const mean=scores.reduce((a,b)=>a+b,0)/scores.length;
    const sd=Math.sqrt(scores.reduce((a,b)=>a+(b-mean)**2,0)/scores.length);
    cells.push(`${(mean*100).toFixed(1)} ± ${(sd*100).toFixed(1)}`);
  }
  rows.push([modelName,...cells]);
}
const widths=[Math.max(15,...rows.map(r=>r[0].length)),...datasets.map(d=>Math.max(d.length,12))];
const format=row=>row.map((v,i)=>String(v).padEnd(widths[i])).join(' | ');
console.log(`Accuracy mean ± standard deviation (%), ${seeds} seeds; train=${trainSize}, test=${testSize}`);
console.log(format(['Classifier',...datasets]));
console.log(widths.map(w=>'-'.repeat(w)).join('-|-'));
rows.forEach(r=>console.log(format(r)));
