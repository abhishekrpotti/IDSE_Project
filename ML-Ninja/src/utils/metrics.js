/**
 * Evaluation Metrics for Classifiers
 */

export function calculateMetrics(yTrue, yPred) {
  let tp = 0, fp = 0, tn = 0, fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const trueVal = yTrue[i] >= 0 ? 1 : -1;
    const predVal = yPred[i] >= 0 ? 1 : -1;

    if (trueVal === 1 && predVal === 1) tp++;
    else if (trueVal === -1 && predVal === 1) fp++;
    else if (trueVal === -1 && predVal === -1) tn++;
    else if (trueVal === 1 && predVal === -1) fn++;
  }

  const total = yTrue.length || 1;
  const accuracy = (tp + tn) / total;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    accuracy,
    precision,
    recall,
    f1,
    tp,
    fp,
    tn,
    fn,
    confusionMatrix: [
      [tn, fp],
      [fn, tp]
    ]
  };
}

