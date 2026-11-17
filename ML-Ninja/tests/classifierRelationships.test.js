import {describe,it,expect} from 'vitest';
import {datasets,makeData} from '../src/data/generators.js';
import {classifierNames,trainClassifier} from '../src/classifiers/index.js';

const accuracy=(model,points)=>points.reduce((n,p)=>n+Number(model.predict(p.x)===p.y),0)/points.length;
function scores(name,dataset,seeds=10){return Array.from({length:seeds},(_,i)=>{const seed=i+1,train=makeData(dataset,100,seed),test=makeData(dataset,120,seed+1000);return accuracy(trainClassifier(name,train,{},seed),test)})}
const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
const variance=values=>{const m=mean(values);return values.reduce((s,v)=>s+(v-m)**2,0)/values.length};

describe('classifier and dataset behavior',()=>{
  it('generates all ten seeded datasets reproducibly',()=>{
    expect(datasets).toHaveLength(10);
    for(const dataset of datasets)expect(makeData(dataset,40,91)).toEqual(makeData(dataset,40,91));
    expect(makeData('Checkerboard (3x3)',40,91)).not.toEqual(makeData('Checkerboard (3x3)',40,92));
  });
  it('keeps linear methods strong on separable blobs and weak on nonlinear layouts',()=>{
    for(const name of ['Perceptron','Logistic Regression','Linear SVM','LDA']){
      expect(mean(scores(name,'Separable blobs'))).toBeGreaterThan(.9);
      for(const dataset of ['XOR','Circles','Checkerboard (3x3)'])expect(mean(scores(name,dataset))).toBeLessThan(.68);
    }
  });
  it('lets nonlinear and tree models learn moons, circles, and checkerboards',()=>{
    for(const name of ['Kernel SVM','k-NN','Decision Tree','Random Forest'])expect(mean(scores(name,'Two moons'))).toBeGreaterThan(.85);
    for(const name of ['Kernel SVM','k-NN'])expect(mean(scores(name,'Circles'))).toBeGreaterThan(.85);
    expect(mean(scores('Decision Tree','Checkerboard (3x3)'))).toBeGreaterThan(.75);expect(mean(scores('Decision Tree','Checkerboard (3x3)'))-mean(scores('Logistic Regression','Checkerboard (3x3)'))).toBeGreaterThan(.15);
    expect(mean(scores('Random Forest','Checkerboard (3x3)'))).toBeGreaterThan(.75);
  });
  it('shows covariance assumptions and ensemble stability',()=>{
    expect(mean(scores('LDA','Correlated blobs'))-mean(scores('Gaussian Naive Bayes','Correlated blobs'))).toBeGreaterThan(.08);
    expect(mean(scores('QDA','Unequal-variance blobs'))-mean(scores('LDA','Unequal-variance blobs'))).toBeGreaterThan(.05);
    for(const dataset of ['Two moons','Overlapping blobs']){const repeat=name=>Array.from({length:20},(_,i)=>{const seed=i+1,train=makeData(dataset,120,seed,{noise:.2}),test=makeData(dataset,180,seed+10000,{noise:.2});return accuracy(trainClassifier(name,train,{},seed),test)});expect(variance(repeat('Random Forest'))-variance(repeat('Decision Tree'))).toBeLessThan(0)}
  });
  it('keeps model predictions distinct across multiple patterns',()=>{
    const patterns=['XOR','Two moons','Correlated blobs','Checkerboard (3x3)'];
    const points=patterns.map((dataset,i)=>makeData(dataset,90,500+i));
    const disagreement=patterns.map(()=>({different:0,total:0}));
    for(let i=0;i<classifierNames.length;i++)for(let j=i+1;j<classifierNames.length;j++){
      let differs=false;
      points.forEach((test,index)=>{const dataset=patterns[index],training=makeData(dataset,100,700+index),a=trainClassifier(classifierNames[i],training,{},701+index),b=trainClassifier(classifierNames[j],training,{},701+index);const count=test.reduce((n,point)=>n+Number(a.predict(point.x)!==b.predict(point.x)),0);disagreement[index].different+=count;disagreement[index].total+=test.length;differs ||= count>0});
      expect(differs).toBe(true);
    }
    expect(disagreement.filter(d=>d.different/d.total>.02)).toHaveLength(4);
  });
});
