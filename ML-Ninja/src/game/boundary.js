export function linearBoundaryClass(a,b,point){
  const side=(b.x-a.x)*(point.y-a.y)-(b.y-a.y)*(point.x-a.x);
  return side>=0?1:-1;
}

/** Classify against freehand paths: closed paths mark their inside; open curves split by their nearest segment. */
export function freehandBoundaryClass(strokes,point){
  let inside=false,nearest=null;
  for(const stroke of strokes??[]){
    if(!stroke||stroke.length<2)continue;
    const first=stroke[0],last=stroke.at(-1);
    const closed=stroke.length>=3&&Math.hypot(last.x-first.x,last.y-first.y)<=.24;
    if(closed){
      for(let i=0,j=stroke.length-1;i<stroke.length;j=i++){
        const a=stroke[i],b=stroke[j];
        if(((a.y>point.y)!==(b.y>point.y))&&(point.x<(b.x-a.x)*(point.y-a.y)/(b.y-a.y)+a.x))inside=!inside;
      }
      continue;
    }
    for(let i=1;i<stroke.length;i++){
      const a=stroke[i-1],b=stroke[i],dx=b.x-a.x,dy=b.y-a.y,length2=dx*dx+dy*dy;
      if(length2===0)continue;
      const t=Math.max(0,Math.min(1,((point.x-a.x)*dx+(point.y-a.y)*dy)/length2));
      const distance=Math.hypot(point.x-(a.x+t*dx),point.y-(a.y+t*dy));
      if(!nearest||distance<nearest.distance)nearest={distance,side:dx*(point.y-a.y)-dy*(point.x-a.x)};
    }
  }
  if(inside)return 1;
  if(nearest)return nearest.side>=0?1:-1;
  return -1;
}
