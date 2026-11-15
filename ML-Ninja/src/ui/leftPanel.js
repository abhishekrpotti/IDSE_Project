import GUI from 'lil-gui';
import { CLASSIFIERS } from '../classifiers/index.js';

export class LeftPanel {
  constructor(container, params, callbacks = {}) {
    this.params = params;
    this.callbacks = callbacks;

    this.gui = new GUI({ container, title: '⚡ ML NINJA CONTROLS' });
    this.gui.domElement.id = 'left-gui';

    // 1. Game Mode Folder
    const modeFolder = this.gui.addFolder('Game Mode');
    modeFolder.add(this.params, 'mode', ['FLOW', 'FREEZE', 'TRAIN', 'TEST', 'CHALLENGE'])
      .name('Mode')
      .onChange(val => callbacks.onModeChange && callbacks.onModeChange(val))
      .listen();

    modeFolder.add({ freeze: () => callbacks.onAction && callbacks.onAction('freeze') }, 'freeze')
      .name('❄️ Freeze (SPACE)');

    modeFolder.add({ train: () => callbacks.onAction && callbacks.onAction('train') }, 'train')
      .name('🧠 Train Model (T)');

    modeFolder.add({ test: () => callbacks.onAction && callbacks.onAction('test') }, 'test')
      .name('🎯 Test (Auto-Slice)');

    modeFolder.add({ challenge: () => callbacks.onAction && callbacks.onAction('challenge') }, 'challenge')
      .name('⚔️ Player vs Model');

    modeFolder.add({ reset: () => callbacks.onAction && callbacks.onAction('reset') }, 'reset')
      .name('↺ Reset All (R)');

    // 2. Dataset Folder
    const dataFolder = this.gui.addFolder('Dataset & Feature Space');
    dataFolder.add(this.params, 'dataset', ['separable', 'overlapping', 'xor', 'moons', 'circles', 'spirals'])
      .name('Distribution')
      .onChange(val => callbacks.onDatasetChange && callbacks.onDatasetChange(val));

    dataFolder.add(this.params, 'noise', 0.0, 0.6, 0.02)
      .name('Noise Level')
      .onChange(() => callbacks.onDatasetChange && callbacks.onDatasetChange(this.params.dataset));

    dataFolder.add(this.params, 'showGrid')
      .name('Feature Grid Plane')
      .onChange(val => callbacks.onToggleGrid && callbacks.onToggleGrid(val));

    dataFolder.add({ orbit: () => callbacks.onAction && callbacks.onAction('orbit') }, 'orbit')
      .name('Inspect 3D (O)');

    // 3. Classifier Folder
    const modelFolder = this.gui.addFolder('Classifier');
    const classifierOptions = {};
    for (const [key, item] of Object.entries(CLASSIFIERS)) {
      classifierOptions[item.name] = key;
    }
    modelFolder.add(this.params, 'classifier', classifierOptions)
      .name('Algorithm')
      .onChange(val => {
        this.updateHyperparamsVisibility(val);
        callbacks.onClassifierChange && callbacks.onClassifierChange(val);
      });

    // 4. Dynamic Hyperparameter Folder
    this.paramFolder = this.gui.addFolder('Hyperparameters');
    this.controllers = {};

    this.controllers.lr = this.paramFolder.add(this.params, 'lr', 0.01, 1.0, 0.01).name('Learning Rate (η)');
    this.controllers.epochs = this.paramFolder.add(this.params, 'epochs', 5, 200, 5).name('Epochs');
    this.controllers.C = this.paramFolder.add(this.params, 'C', 0.1, 10.0, 0.1).name('Regularization (C)');
    this.controllers.gamma = this.paramFolder.add(this.params, 'gamma', 0.1, 3.0, 0.1).name('RBF Gamma (γ)');
    this.controllers.k = this.paramFolder.add(this.params, 'k', 1, 15, 2).name('Neighbors (k)');
    this.controllers.metric = this.paramFolder.add(this.params, 'metric', ['euclidean', 'manhattan']).name('Distance Metric');
    this.controllers.maxDepth = this.paramFolder.add(this.params, 'maxDepth', 1, 8, 1).name('Max Depth');
    this.controllers.nTrees = this.paramFolder.add(this.params, 'nTrees', 2, 20, 1).name('Num Trees');

    this.updateHyperparamsVisibility(this.params.classifier);
  }

  updateHyperparamsVisibility(classifierId) {
    // Hide all first
    Object.values(this.controllers).forEach(c => c.hide());

    if (classifierId === 'perceptron') {
      this.controllers.lr.show();
      this.controllers.epochs.show();
    } else if (classifierId === 'logistic') {
      this.controllers.lr.show();
      this.controllers.epochs.show();
    } else if (classifierId === 'svm') {
      this.controllers.C.show();
      this.controllers.lr.show();
      this.controllers.epochs.show();
    } else if (classifierId === 'kernelSvm') {
      this.controllers.C.show();
      this.controllers.gamma.show();
      this.controllers.epochs.show();
    } else if (classifierId === 'knn') {
      this.controllers.k.show();
      this.controllers.metric.show();
    } else if (classifierId === 'tree') {
      this.controllers.maxDepth.show();
    } else if (classifierId === 'forest') {
      this.controllers.nTrees.show();
      this.controllers.maxDepth.show();
    } else if (classifierId === 'mlp') {
      this.controllers.lr.show();
      this.controllers.epochs.show();
    }
  }
}

