export class TutorialOverlay {
  constructor(container, onDismiss) {
    this.container = container;
    this.onDismiss = onDismiss;
    this.currentStep = 0;

    this.steps = [
      {
        title: '🗡️ Slice Fruits, Avoid Bombs!',
        body: 'Every flying object is a DATA POINT in 2D feature space. Juicy fruits are <b>Class +1</b> and explosive bombs are <b>Class -1</b>. Drag your mouse to slice!',
        actionText: 'Next: Feature Space →'
      },
      {
        title: '❄️ Press SPACE to Freeze Time',
        body: 'Hit <b>SPACE</b> at any moment to freeze time! The objects settle into a static labeled scatter plot on the 2D plane. This becomes your ML <b>Training Dataset</b>.',
        actionText: 'Next: Train Models →'
      },
      {
        title: '🧠 Train & Compare Classifiers',
        body: 'Select any of the 10 machine learning classifiers and press <b>T</b> to train step-by-step. Watch decision boundaries rotate, bend, and separate! In <b>Test mode</b>, watch auto-pilot slice predicted fruits.',
        actionText: 'Start Playing ML Ninja ⚔️'
      }
    ];

    this.element = document.createElement('div');
    this.element.id = 'tutorial-modal';
    this.element.className = 'tutorial-backdrop';
    this.render();
    container.appendChild(this.element);
  }

  render() {
    const step = this.steps[this.currentStep];
    this.element.innerHTML = `
      <div class="tutorial-card glass-panel">
        <div class="tutorial-step-indicator">STEP ${this.currentStep + 1} OF ${this.steps.length}</div>
        <h2 class="tutorial-title">${step.title}</h2>
        <p class="tutorial-body">${step.body}</p>
        <div class="tutorial-actions">
          <button id="btn-tutorial-skip" class="ctrl-btn">Skip Tutorial</button>
          <button id="btn-tutorial-next" class="ctrl-btn primary-ctrl">${step.actionText}</button>
        </div>
      </div>
    `;

    this.element.querySelector('#btn-tutorial-next').addEventListener('click', () => {
      this.currentStep++;
      if (this.currentStep >= this.steps.length) {
        this.close();
      } else {
        this.render();
      }
    });

    this.element.querySelector('#btn-tutorial-skip').addEventListener('click', () => {
      this.close();
    });
  }

  close() {
    this.element.style.display = 'none';
    if (this.onDismiss) this.onDismiss();
  }
}

