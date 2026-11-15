import { CLASSIFIERS } from '../classifiers/index.js';

export class RightPanel {
  constructor(container) {
    this.element = document.createElement('div');
    this.element.id = 'right-panel';
    this.element.className = 'glass-panel';
    this.element.innerHTML = `
      <div class="panel-header">
        <h3 id="explainer-title">Perceptron</h3>
        <span class="badge-tag" id="explainer-family">Linear</span>
      </div>
      <div class="explainer-section">
        <div class="section-label">KEY FORMULA</div>
        <div class="formula-box" id="explainer-formula">w · x + b = 0</div>
      </div>
      <div class="explainer-section">
        <div class="section-label">HOW IT LEARNS</div>
        <p id="explainer-summary" class="section-text"></p>
      </div>
      <div class="explainer-section">
        <div class="section-label failure-label">WHEN IT FAILS</div>
        <p id="explainer-fails" class="section-text failure-text"></p>
      </div>
      <div class="explainer-section" id="live-step-section">
        <div class="section-label">LIVE TRAINING STEP</div>
        <div class="step-metrics" id="explainer-step-info">
          <div>Step: <span id="metric-step">0</span></div>
          <div>Status: <span id="metric-status">Ready</span></div>
        </div>
      </div>
    `;
    container.appendChild(this.element);

    this.titleEl = this.element.querySelector('#explainer-title');
    this.familyEl = this.element.querySelector('#explainer-family');
    this.formulaEl = this.element.querySelector('#explainer-formula');
    this.summaryEl = this.element.querySelector('#explainer-summary');
    this.failsEl = this.element.querySelector('#explainer-fails');
    this.stepEl = this.element.querySelector('#metric-step');
    this.statusEl = this.element.querySelector('#metric-status');
  }

  setClassifier(classifierId) {
    const info = CLASSIFIERS[classifierId];
    if (!info) return;

    this.titleEl.textContent = info.name;
    this.formulaEl.textContent = info.formula;
    this.summaryEl.textContent = info.summary;
    this.failsEl.textContent = info.whenItFails;

    const isLinear = ['perceptron', 'logistic', 'svm', 'lda'].includes(classifierId);
    this.familyEl.textContent = isLinear ? 'Linear Model' : 'Non-Linear / Non-Parametric';
    this.familyEl.className = `badge-tag ${isLinear ? 'badge-linear' : 'badge-nonlinear'}`;

    this.stepEl.textContent = '0';
    this.statusEl.textContent = 'Idle';
  }

  updateStepInfo(stepData) {
    if (!stepData) return;
    this.stepEl.textContent = stepData.step ?? 0;
    this.statusEl.textContent = stepData.type === 'complete' ? 'Converged / Done' : (stepData.phase || `Iterating (Loss: ${stepData.loss?.toFixed(4) ?? '--'})`);
  }
}

