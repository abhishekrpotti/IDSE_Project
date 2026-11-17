import {accuracy} from '../utils/metrics.js';
import {finalizeRound} from './roundStats.js';

// Kept as a small adapter for callers that only need evaluation, not round scoring.
export function evaluate(data, model, player) {
  const truth = data.map(point => point.y);
  const predictions = data.map(point => model.predict(point.x));
  const modelAccuracy = accuracy(truth, predictions);
  let best = { accuracy: 0, flipped: false };
  for (const flipped of [false, true]) {
    const score = accuracy(truth, data.map(point => player(point.x, flipped)));
    if (score > best.accuracy) best = { accuracy: score, flipped };
  }
  const verdict = best.accuracy >= 0.9 || best.accuracy >= modelAccuracy - 0.05
    ? 'Nice cut!'
    : best.accuracy >= 0.75 ? 'Almost there!' : 'Missed the mark';
  return { ...best, modelAccuracy, verdict, predictions };
}

export {finalizeRound};
