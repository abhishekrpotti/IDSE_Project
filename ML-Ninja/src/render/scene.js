import * as THREE from 'three';
import {theme} from '../theme.js';

const haloGeo=new THREE.CircleGeometry(.34,24);
const fruitTextures=theme.fruit.map((color,index)=>makeFruitTexture(color,index));
const bombTexture=makeBombTexture();
const haloMaterials={fruit:new THREE.MeshBasicMaterial({color:theme.fruitHalo,transparent:true,opacity:.55}),bomb:new THREE.MeshBasicMaterial({color:theme.bombHalo,transparent:true,opacity:.62})};
const spriteMaterials={fruit:fruitTextures.map(map=>new THREE.SpriteMaterial({map,transparent:true,depthTest:false})),bomb:new THREE.SpriteMaterial({map:bombTexture,transparent:true,depthTest:false})};

export function createScene(canvas){
  const scene=new THREE.Scene();
  const initialRect=canvas.getBoundingClientRect();
  const initialWidth=initialRect.width||innerWidth,initialHeight=initialRect.height||innerHeight;
  const camera=new THREE.OrthographicCamera(-(initialWidth/initialHeight)*3.2,(initialWidth/initialHeight)*3.2,3.2,-3.2,.1,100);
  camera.position.set(0,0,12);
  const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.5));
  renderer.setSize(initialWidth,initialHeight,false);
  const objects=[];
  function makeFruit(data,index){
    const group=new THREE.Group();
    const kind=data.fruit?'fruit':'bomb';
    const halo=new THREE.Mesh(haloGeo,haloMaterials[kind]);
    halo.position.z=-.2;halo.scale.set(1.45,1.45,1);group.add(halo);
    const sprite=new THREE.Sprite(data.fruit?spriteMaterials.fruit[index%spriteMaterials.fruit.length]:spriteMaterials.bomb);
    sprite.scale.set(.82,.82,1);sprite.position.z=.15;sprite.userData.hitShape=true;group.add(sprite);
    scene.add(group);
    group.userData={...data,halo,sprite,alive:true};objects.push(group);
    return group;
  }
  function render(){renderer.render(scene,camera)}
  function resize(){const rect=canvas.getBoundingClientRect(),w=Math.max(rect.width,1),h=Math.max(rect.height,1);camera.left=-w/h*3.2;camera.right=w/h*3.2;camera.top=3.2;camera.bottom=-3.2;camera.updateProjectionMatrix();renderer.setSize(w,h,false);render()}
  addEventListener('resize',resize);
  const resizeObserver=new ResizeObserver(resize);resizeObserver.observe(canvas.parentElement);
  return{scene,camera,renderer,objects,makeFruit,render,resize};
}

function makeFruitTexture(color,type){
  const canvas=document.createElement('canvas');canvas.width=128;canvas.height=128;
  const c=canvas.getContext('2d');c.lineWidth=6;c.lineJoin='round';c.strokeStyle=theme.outline;c.fillStyle=color;
  if(type===2){c.beginPath();c.moveTo(28,48);c.quadraticCurveTo(64,34,100,48);c.lineTo(91,78);c.quadraticCurveTo(64,112,37,78);c.closePath();c.fill();c.stroke()}
  else if(type===3){for(const [x,y,r] of [[48,54,18],[75,52,18],[38,77,18],[64,78,19],[88,77,17],[51,99,17],[75,99,17]]){c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill();c.stroke()}}
  else if(type===5){c.beginPath();c.moveTo(64,108);c.bezierCurveTo(43,89,26,67,34,49);c.bezierCurveTo(42,31,56,42,64,48);c.bezierCurveTo(75,38,91,34,98,51);c.bezierCurveTo(108,74,81,99,64,108);c.closePath();c.fill();c.stroke()}
  else {c.beginPath();c.moveTo(64,45);c.bezierCurveTo(52,31,30,39,29,62);c.bezierCurveTo(27,86,47,105,64,108);c.bezierCurveTo(82,105,101,85,99,61);c.bezierCurveTo(97,39,76,31,64,45);c.closePath();c.fill();c.stroke()}
  c.fillStyle='#85b978';c.beginPath();c.ellipse(70,34,15,7,-.55,0,Math.PI*2);c.fill();c.strokeStyle=theme.outline;c.lineWidth=4;c.stroke();c.beginPath();c.moveTo(64,47);c.quadraticCurveTo(65,35,70,28);c.stroke();
  c.fillStyle='#35364c';c.beginPath();c.arc(50,69,4.5,0,Math.PI*2);c.arc(78,69,4.5,0,Math.PI*2);c.fill();c.strokeStyle='#35364c';c.lineWidth=4;c.beginPath();c.arc(64,70,13,.25,Math.PI-.25);c.stroke();
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;return texture;
}
function makeBombTexture(){
  const canvas=document.createElement('canvas');canvas.width=128;canvas.height=128;const c=canvas.getContext('2d');
  c.fillStyle=theme.bomb;c.strokeStyle=theme.outline;c.lineWidth=7;c.beginPath();c.arc(60,76,42,0,Math.PI*2);c.fill();c.stroke();
  c.strokeStyle='#b8a5c8';c.lineWidth=9;c.beginPath();c.moveTo(81,42);c.quadraticCurveTo(94,27,101,15);c.stroke();
  c.fillStyle='#ff8fab';c.beginPath();c.moveTo(102,3);c.lineTo(108,13);c.lineTo(120,13);c.lineTo(112,21);c.lineTo(116,32);c.lineTo(103,27);c.lineTo(94,35);c.lineTo(95,22);c.lineTo(86,16);c.lineTo(99,13);c.closePath();c.fill();
  c.fillStyle='#fff5e9';c.beginPath();c.arc(47,70,4,0,Math.PI*2);c.arc(73,70,4,0,Math.PI*2);c.fill();
  c.strokeStyle='#fff5e9';c.lineWidth=4;c.beginPath();c.moveTo(47,94);c.lineTo(56,88);c.lineTo(64,94);c.lineTo(72,88);c.lineTo(80,94);c.stroke();
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;return texture;
}
