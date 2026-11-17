import {it,expect} from 'vitest';import {evaluate} from '../src/game/evaluation.js';
it('chooses the better orientation of a drawn boundary',()=>{const data=[{x:[0,1],y:1},{x:[1,0],y:-1}];const out=evaluate(data,{predict:x=>x[0]===0?1:-1},(x,flip)=>{const p=x[0]===0?-1:1;return flip?-p:p});expect(out.accuracy).toBe(1);expect(out.flipped).toBe(true);expect(out.verdict).toBe('Nice cut!')});
