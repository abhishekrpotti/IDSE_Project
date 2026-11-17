const DOMAIN=2;
export function drawResultGraph(canvas,stats){
  const ctx=canvas.getContext('2d'),width=canvas.width,height=canvas.height;
  const colorblind=document.body.classList.contains('colorblind-mode');
  ctx.clearRect(0,0,width,height);
  const grid=stats.modelPredictionGrid,size=grid.length;
  for(let row=0;row<size;row++)for(let col=0;col<size;col++){
    const probability=stats.graphOptions?.probability?stats.modelProbabilityGrid?.[row]?.[col]:null;
    const certainty=probability==null?1:Math.min(1,Math.abs(probability-.5)*2);
    const alpha=0.2+certainty*.62;
    ctx.fillStyle=grid[row][col]===1?`rgba(${colorblind?'86,180,233':'142,227,194'},${alpha})`:`rgba(${colorblind?'230,159,0':'255,138,138'},${alpha})`;
    ctx.fillRect(col*width/size,row*height/size,width/size+1,height/size+1);
  }
  const px=x=>(x+DOMAIN)/(DOMAIN*2)*width;
  const py=y=>(DOMAIN-y)/(DOMAIN*2)*height;
  if(stats.classifierName==='Linear SVM'&&stats.modelLinearSvm)drawSvmMargins(ctx,stats.modelLinearSvm,px,py);
  if(stats.classifierName==='Kernel SVM'&&stats.modelScoreGrid?.length)drawKernelMargins(ctx,stats.modelScoreGrid,width,height);
  if(stats.graphOptions?.training&&stats.trainingPoints)for(const point of stats.trainingPoints){ctx.beginPath();ctx.moveTo(px(point.x[0]),py(point.x[1])-3);ctx.lineTo(px(point.x[0])-3,py(point.x[1])+3);ctx.lineTo(px(point.x[0])+3,py(point.x[1])+3);ctx.closePath();ctx.fillStyle=point.y===1?'#147c5b':'#ae4351';ctx.fill()}
  for(const [index,point] of stats.points.entries()){
    ctx.beginPath();
    const x=px(point.x[0]),y=py(point.x[1]);
    if(point.y===1)ctx.arc(x,y,4,0,Math.PI*2);else{ctx.moveTo(x,y-5);ctx.lineTo(x+5,y);ctx.lineTo(x,y+5);ctx.lineTo(x-5,y);ctx.closePath()}
    ctx.fillStyle=point.y===1?(colorblind?'#0072b2':'#4d9b7c'):(colorblind?'#b87500':'#c16a72');ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
    if(stats.modelPredictions&&stats.modelPredictions[index]!==point.y){ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.strokeStyle='#e49b16';ctx.lineWidth=2.5;ctx.stroke()}
  }
  if(['Linear SVM','Kernel SVM'].includes(stats.classifierName))for(const vector of stats.modelSupportVectors??[]){const x=px(vector.x[0]),y=py(vector.x[1]);ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.fillStyle='rgba(255,241,168,.8)';ctx.fill();ctx.strokeStyle='#33264a';ctx.lineWidth=2.5;ctx.stroke();ctx.beginPath();if(vector.y===1)ctx.arc(x,y,4,0,Math.PI*2);else{ctx.moveTo(x,y-4);ctx.lineTo(x+4,y);ctx.lineTo(x,y+4);ctx.lineTo(x-4,y);ctx.closePath()}ctx.fillStyle=vector.y===1?'#4d9b7c':'#c16a72';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke()}else if(stats.graphOptions?.supportVectors)for(const vector of stats.modelSupportVectors??[]){ctx.beginPath();ctx.arc(px(vector.x[0]),py(vector.x[1]),7,0,Math.PI*2);ctx.strokeStyle='#25243a';ctx.lineWidth=2;ctx.stroke()}
  if(stats.graphOptions?.margins&&stats.modelProbabilityGrid?.length){ctx.strokeStyle='rgba(35,35,55,.6)';ctx.lineWidth=1;for(const level of [.35,.65]){for(let row=1;row<size-1;row++)for(let col=1;col<size-1;col++){const p=stats.modelProbabilityGrid[row][col];if(Math.abs(p-level)<.012){ctx.beginPath();ctx.arc((col+.5)*width/size,(row+.5)*height/size,1.1,0,Math.PI*2);ctx.stroke()}}}}
  const boundary=stats.userBoundary;
  if(boundary?.type==='line')drawPath(ctx,boundary.points,px,py,stats.displayTransform);
  else for(const stroke of boundary?.strokes??[])drawPath(ctx,stroke,px,py,stats.displayTransform);
  ctx.strokeStyle='#6d6875';ctx.lineWidth=1;ctx.strokeRect(.5,.5,width-1,height-1);
}

function drawKernelMargins(ctx,grid,width,height){
  const rows=grid.length,cols=grid[0]?.length??0;
  if(rows<2||cols<2)return;
  const drawLevel=(level,color,lineWidth)=>{
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash([]);
    const edges=[[0,1],[1,2],[2,3],[3,0]],isMargin=level!==0;
    for(let row=0;row<rows-1;row++)for(let col=0;col<cols-1;col++){
      // Leave periodic gaps in the tiny cell segments so margins remain visibly dashed.
      if(isMargin&&(row+col)%3!==0)continue;
      const cell=[
        {x:col,y:row,v:grid[row][col]},
        {x:col+1,y:row,v:grid[row][col+1]},
        {x:col+1,y:row+1,v:grid[row+1][col+1]},
        {x:col,y:row+1,v:grid[row+1][col]},
      ],crossings=[];
      for(const[a,b]of edges){
        const p=cell[a],q=cell[b];
        if((p.v<level&&q.v>=level)||(q.v<level&&p.v>=level)){
          const t=(level-p.v)/(q.v-p.v);
          crossings.push({x:p.x+(q.x-p.x)*t,y:p.y+(q.y-p.y)*t});
        }
      }
      for(let i=0;i+1<crossings.length;i+=2){
        const a=crossings[i],b=crossings[i+1];
        ctx.moveTo(a.x/(cols-1)*width,a.y/(rows-1)*height);
        ctx.lineTo(b.x/(cols-1)*width,b.y/(rows-1)*height);
      }
    }
    ctx.stroke();
  };
  drawLevel(-1,'#27334f',1.5);
  drawLevel(1,'#27334f',1.5);
  drawLevel(0,'#302243',2.5);
}
function drawSvmMargins(ctx,{w, b, mean, scale},px,py){const a=w[0]/scale[0],c=w[1]/scale[1],offset=b-a*mean[0]-c*mean[1];const intersections=level=>{const found=[],add=(x,y)=>{if(x>=-DOMAIN-1e-7&&x<=DOMAIN+1e-7&&y>=-DOMAIN-1e-7&&y<=DOMAIN+1e-7&&!found.some(p=>Math.hypot(p.x-x,p.y-y)<1e-5))found.push({x,y})};if(Math.abs(c)>1e-9)for(const x of [-DOMAIN,DOMAIN])add(x,(level-offset-a*x)/c);if(Math.abs(a)>1e-9)for(const y of [-DOMAIN,DOMAIN])add((level-offset-c*y)/a,y);return found};for(const level of [-1,1]){const ends=intersections(level);if(ends.length<2)continue;ctx.beginPath();ctx.moveTo(px(ends[0].x),py(ends[0].y));ctx.lineTo(px(ends[1].x),py(ends[1].y));ctx.strokeStyle='#27334f';ctx.lineWidth=1.6;ctx.setLineDash([6,5]);ctx.stroke()}ctx.setLineDash([]);const center=intersections(0);if(center.length>=2){ctx.beginPath();ctx.moveTo(px(center[0].x),py(center[0].y));ctx.lineTo(px(center[1].x),py(center[1].y));ctx.strokeStyle='#302243';ctx.lineWidth=2.5;ctx.stroke()}}
function drawPath(ctx,points,px,py,transform={cx:0,cy:0,sx:1.6,sy:1.05}){
  if(!points||points.length<2)return;
  ctx.beginPath();ctx.moveTo(px(points[0].x/transform.sx+transform.cx),py(points[0].y/transform.sy+transform.cy));
  for(const point of points.slice(1))ctx.lineTo(px(point.x/transform.sx+transform.cx),py(point.y/transform.sy+transform.cy));
  ctx.strokeStyle='#fff';ctx.lineWidth=3.5;ctx.lineCap='round';ctx.lineJoin='round';ctx.stroke();
}
