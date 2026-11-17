const TOUR_STEPS = [
  {
    icon: '🥷',
    section: 'START HERE',
    title: 'Welcome to your ML experiment',
    body: 'ML Ninja turns classification into a game. Each fruit or bomb is an example with two position features. Fruit is class +1; bombs are class −1. Your job is to draw a boundary that predicts the labels.',
    location: 'The center play field is the dataset shown as moving objects.',
    points: [
      'A model learns a rule from known examples, then predicts labels for new points.',
      'There is no single best classifier for every pattern.',
    ],
  },
  {
    icon: '🧠',
    section: 'YOUR LAB · MODEL',
    title: 'Choose a classifier',
    body: 'The Model section on the left lets you pick the learning method and adjust the controls that method supports. The short explanation changes with your choice.',
    location: 'Your lab → Model. On a narrow screen, tap Lab to open it.',
    points: [
      'Start with a simple model, then compare it with methods that can draw curved or more detailed regions.',
      'Challenge mode changes what you are asked to predict. Guided lesson loads a short model-and-dataset activity.',
      'Use the model’s assumptions to explain why it succeeds or struggles.',
    ],
  },
  {
    icon: '🧪',
    section: 'YOUR LAB · DATA',
    title: 'Understand the data first',
    body: 'Choose a dataset pattern before choosing a complicated model. The “About this dataset” note describes its structure and what makes it different.',
    location: 'Your lab → Data, then Wave settings.',
    points: [
      'The size readout tells you how many fruit and bomb examples are in the upcoming wave.',
      'The classifier learns from 100 separate training examples; the wave is held out to check its predictions.',
      'Wave size changes how many examples are scored. Feature noise spreads points; label flips deliberately change some labels.',
      'Change one setting at a time so you can explain what caused a boundary or score to change.',
    ],
  },
  {
    icon: '🎨',
    section: 'YOUR LAB · DISPLAY',
    title: 'Make the game comfortable to use',
    body: 'Display controls adjust how the game looks and moves. They do not change the classifier or its predictions.',
    location: 'Your lab → Display.',
    points: [
      'Colorblind palette changes the plot colors and point shapes.',
      'Reduce motion shortens visual transitions; Low quality can help on slower devices.',
      'The game uses generated synthetic data. It is useful for learning boundaries, but it does not demonstrate sourcing or cleaning a real-world dataset.',
    ],
  },
  {
    icon: '🍉',
    section: 'PLAY FIELD',
    title: 'Draw, slice, and submit',
    body: 'Start a wave with the button or Space. Watch the points settle, then use the 10-second drawing window to show how you would separate the classes.',
    location: 'The center of the screen; the bottom buttons start, submit, switch to Slice mode, and control sound.',
    points: [
      'Perceptron, Logistic Regression, and Linear SVM use a straight line. Other models let you draw a freehand boundary.',
      'In Slice mode, swipe through objects to hit fruit and avoid bombs. Press Submit or Enter to finish early.',
      'Fruit hits add score, bomb hits subtract score, and agreement with the trained model adds a bonus.',
    ],
  },
  {
    icon: '💡',
    section: 'HOW IT WORKS',
    title: 'Use the live explanation',
    body: 'The right panel follows your selected classifier and dataset. It explains the model’s idea, the data pattern, and one experiment you can try.',
    location: 'How it works on the right; on a narrow screen, tap the How it works button.',
    points: [
      'The small plot previews the model’s predicted regions for the current settings.',
      'Read “When it struggles” to understand which assumptions may not fit your data.',
      'Use the suggested experiment to compare a specific pair of models or settings.',
    ],
  },
  {
    icon: '📊',
    section: 'ROUND RESULTS',
    title: 'Explain the result, not just the score',
    body: 'Your accuracy evaluates the boundary you drew. Model accuracy evaluates the classifier on this wave after it learned from separate training examples. The confusion matrix shows the model’s correct predictions and mistakes.',
    location: 'The results card appears when the round ends.',
    points: [
      'Precision, recall, and F1 describe fruit predictions from different angles; accuracy alone can hide which class is being missed.',
      'Inspect the decision-region graph and the points marked as mistakes. Compare all classifiers to see how their boundaries differ on the same wave.',
      'A wave is small, so one point can noticeably change its percentage. Repeat the experiment before making a broad claim.',
      'Round score is a game score. It includes slicing and boundary agreement, so it is not the same thing as model accuracy.',
    ],
  },
  {
    icon: '📝',
    section: 'MINI PROJECT CHECKLIST',
    title: 'Turn play into sound analysis',
    body: 'A strong data-science project earns credit for justified choices and interpretation, not for using the most complex model. Try this sequence and record what you observe.',
    location: 'Use this checklist after a few rounds or before presenting your findings.',
    points: [
      'Ask a measurable question, such as: “Which classifier best separates the two moons at this noise level?” Explain why this dataset is suitable for that question.',
      'Compare a simple baseline with at least two alternatives on the same dataset and settings. Use Compare all, then inspect the confusion matrix and errors.',
      'Report the wave size, noise, label flips, classifier, and hyperparameters. Retry same data to repeat the current wave.',
      'Discuss limitations and next steps. This game uses synthetic data and a simple holdout wave, not cross-validation. A full real-data project still needs a justified source, preparation/EDA, missing-value or duplicate handling, and a validation plan.',
      'Before presenting, be ready to explain your question, why you chose the data and methods, what evidence supports your conclusion, and what you would investigate next.',
    ],
  },
];

const escapeHtml = (value) => value.replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[character]));

export function shouldShowGameGuide() {
  try {
    return localStorage.getItem('mlninja.guideSeen') !== 'true';
  } catch {
    return true;
  }
}

export function openGameGuide(container) {
  let activeStep = 0;
  let isOpen = true;

  const close = () => {
    if (!isOpen) return;
    isOpen = false;
    container.innerHTML = '';
    document.removeEventListener('keydown', handleKeydown);
    try { localStorage.setItem('mlninja.guideSeen', 'true'); } catch { /* storage can be disabled */ }
    document.querySelector('#guideButton')?.focus();
  };

  const render = () => {
    const step = TOUR_STEPS[activeStep];
    const progress = Math.round(((activeStep + 1) / TOUR_STEPS.length) * 100);
    container.innerHTML = `<div class="overlay guide-overlay"><section class="guide-dialog" role="dialog" aria-modal="true" aria-labelledby="gameGuideTitle" aria-describedby="gameGuideBody"><div class="guide-heading"><span>ML NINJA · QUICK GUIDE</span><button type="button" class="guide-close" data-guide-close aria-label="Close guide">×</button></div><div class="guide-progress"><div class="guide-progress-label"><span>${escapeHtml(step.section)}</span><span>Step ${activeStep + 1} of ${TOUR_STEPS.length}</span></div><div class="guide-progress-track" role="progressbar" aria-label="Guide progress" aria-valuemin="1" aria-valuemax="${TOUR_STEPS.length}" aria-valuenow="${activeStep + 1}"><span style="width:${progress}%"></span></div></div><div class="guide-content"><div class="guide-icon" aria-hidden="true">${step.icon}</div><h1 id="gameGuideTitle">${escapeHtml(step.title)}</h1><p id="gameGuideBody" class="guide-body-copy">${escapeHtml(step.body)}</p><div class="guide-location"><b>Where to look</b><span>${escapeHtml(step.location)}</span></div><ul>${step.points.map((point) => `<li>${escapeHtml(point)}</li>`).join('')}</ul></div><footer class="guide-actions"><button type="button" data-guide-skip class="guide-skip">Skip guide</button><div class="guide-nav">${activeStep > 0 ? '<button type="button" data-guide-back>Back</button>' : ''}<button type="button" class="primary" data-guide-next>${activeStep === TOUR_STEPS.length - 1 ? 'Start exploring' : 'Next'}</button></div></footer></section></div>`;

    container.querySelector('[data-guide-close]').addEventListener('click', close);
    container.querySelector('[data-guide-skip]').addEventListener('click', close);
    container.querySelector('[data-guide-back]')?.addEventListener('click', () => {
      activeStep = Math.max(0, activeStep - 1);
      render();
    });
    container.querySelector('[data-guide-next]').addEventListener('click', () => {
      if (activeStep === TOUR_STEPS.length - 1) {
        close();
        return;
      }
      activeStep += 1;
      render();
    });
    container.querySelector('.guide-overlay').addEventListener('click', (event) => {
      if (event.target === event.currentTarget) close();
    });
    container.querySelector('[data-guide-next]').focus();
  };

  function handleKeydown(event) {
    if (!isOpen) return;
    event.stopPropagation();
    if (event.key === 'Tab') {
      const controls = [...container.querySelectorAll('.guide-dialog button:not(:disabled)')];
      const first = controls[0];
      const last = controls.at(-1);
      if (!controls.includes(document.activeElement)) {
        event.preventDefault();
        first?.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    } else if (event.key === 'Escape') {
      event.preventDefault();
      close();
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      container.querySelector('[data-guide-next]')?.click();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      container.querySelector('[data-guide-back]')?.click();
    }
  }

  document.addEventListener('keydown', handleKeydown);
  render();
}
