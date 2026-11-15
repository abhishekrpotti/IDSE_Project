import * as THREE from 'three';
import { createSceneSetup } from './render/scene.js';
import { createPostProcessing } from './render/postprocessing.js';
import { BladeTrail } from './render/bladeTrail.js';
import { BoundaryMesh } from './render/boundaryMesh.js';
import { ParticleSystem } from './render/particles.js';

import { PhysicsSystem } from './game/physics.js';
import { SlicingController } from './game/slicing.js';
import { Spawner } from './game/spawner.js';
import { ScoringSystem } from './game/scoring.js';
import { sounds } from './audio/soundEffects.js';

import { CLASSIFIERS } from './classifiers/index.js';
import { calculateMetrics } from './utils/metrics.js';

import { HUD } from './ui/hud.js';
import { LeftPanel } from './ui/leftPanel.js';
import { RightPanel } from './ui/rightPanel.js';
import { ChartPanels } from './ui/charts.js';
import { BottomControls } from './ui/bottomControls.js';
import { TooltipController } from './ui/tooltips.js';
import { TutorialOverlay } from './ui/tutorial.js';

class MLNinjaApp {
  constructor() {
    this.container = document.getElementById('canvas-container');
    const { scene, camera, renderer, controls, toggleOrbit, handleResize } = createSceneSetup(this.container);
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.controls = controls;
    this.toggleOrbit = toggleOrbit;
    this.handleResize = handleResize;

    // Post processing for glowing trails
    this.postProcessing = createPostProcessing(renderer, scene, camera);

    // Systems
    this.physics = new PhysicsSystem();
    this.particles = new ParticleSystem(scene, 800);
    this.bladeTrail = new BladeTrail(scene, 24);
    this.boundaryMesh = new BoundaryMesh(scene);
    this.scoring = new ScoringSystem();
    this.slicing = new SlicingController(camera, 0);
    this.spawner = new Spawner(scene, this.particles, sounds);

    // State parameters
    this.params = {
      mode: 'FLOW', // FLOW, FREEZE, TRAIN, TEST, CHALLENGE
      dataset: 'separable',
      noise: 0.15,
      showGrid: true,
      classifier: 'perceptron',
      lr: 0.1,
      epochs: 50,
      C: 1.0,
      gamma: 0.6,
      k: 5,
      metric: 'euclidean',
      maxDepth: 4,
      nTrees: 6
    };

    // ML State
    this.currentClassifier = null;
    this.trainingGenerator = null;
    this.isTrainingPlaying = false;
    this.trainingSpeed = 20; // iterations per second
    this.lastTrainingStepTime = 0;
    this.trainingData = { X: [], y: [] };

    // UI elements
    this.hud = new HUD(document.body);
    this.rightPanel = new RightPanel(document.body);
    this.charts = new ChartPanels(document.body);
    this.tooltips = new TooltipController(document.body);
    this.bottomControls = new BottomControls(document.body, {
      onTogglePlay: (playing) => { this.isTrainingPlaying = playing; },
      onStep: () => { this.stepTraining(); },
      onReset: () => { this.resetTraining(); },
      onSpeedChange: (speed) => { this.trainingSpeed = speed; }
    });

    this.leftPanel = new LeftPanel(
      document.getElementById('left-gui-container'),
      this.params,
      {
        onModeChange: (m) => this.setMode(m),
        onDatasetChange: (d) => this.changeDataset(d),
        onClassifierChange: (c) => this.changeClassifier(c),
        onToggleGrid: (show) => this.boundaryMesh.setVisible(show),
        onAction: (act) => this.handleAction(act)
      }
    );

    // Tutorial overlay
    this.tutorial = new TutorialOverlay(document.body, () => {
      sounds.playClick();
    });

    this.initClassifier();
    this.bindEvents();
    this.lastTime = performance.now();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  initClassifier() {
    const config = CLASSIFIERS[this.params.classifier];
    if (!config) return;
    this.currentClassifier = new config.Class({
      lr: this.params.lr,
      epochs: this.params.epochs,
      C: this.params.C,
      gamma: this.params.gamma,
      k: this.params.k,
      metric: this.params.metric,
      maxDepth: this.params.maxDepth,
      nTrees: this.params.nTrees
    });
    this.rightPanel.setClassifier(this.params.classifier);
  }

  setMode(mode) {
    this.params.mode = mode;
    this.hud.update({ mode });

    if (mode === 'FLOW') {
      this.boundaryMesh.clearOverlay();
    } else if (mode === 'FREEZE') {
      sounds.playFreeze();
      this.gatherFrozenTrainingData();
    } else if (mode === 'TRAIN') {
      this.startTraining();
    } else if (mode === 'TEST') {
      // Clear frozen objects and throw new ones to test
      this.spawner.clearAll();
      sounds.playSuccess();
    } else if (mode === 'CHALLENGE') {
      sounds.playClick();
    }
  }

  gatherFrozenTrainingData() {
    // Collect dataset from active objects or spawner's dataset
    if (this.spawner.activeObjects.length >= 8) {
      this.trainingData.X = this.spawner.activeObjects.map(o => [o.position.x, o.position.y]);
      this.trainingData.y = this.spawner.activeObjects.map(o => o.userData.label);
    } else {
      // Use dataset generator points
      this.trainingData.X = [...this.spawner.dataset.X];
      this.trainingData.y = [...this.spawner.dataset.y];
      // Spawn them all frozen
      this.spawner.clearAll();
      for (let i = 0; i < this.trainingData.X.length; i++) {
        const pt = this.trainingData.X[i];
        const lbl = this.trainingData.y[i];
        const obj = this.spawner.spawnSinglePoint(pt[0], pt[1], lbl);
        obj.position.set(pt[0], pt[1], 0);
        obj.userData.physics.isFrozen = true;
      }
    }
  }

  startTraining() {
    this.gatherFrozenTrainingData();
    this.initClassifier();
    this.charts.reset();
    this.trainingGenerator = this.currentClassifier.fitSteps(this.trainingData.X, this.trainingData.y);
    this.isTrainingPlaying = true;
    this.bottomControls.setPlaying(true);
  }

  stepTraining() {
    if (!this.trainingGenerator) {
      this.startTraining();
    }
    const result = this.trainingGenerator.next();
    if (result.done) {
      this.isTrainingPlaying = false;
      this.bottomControls.setPlaying(false);
      sounds.playSuccess();
      return false;
    }

    const stepData = result.value;
    this.rightPanel.updateStepInfo(stepData);

    // Update boundary texture & overlays
    this.boundaryMesh.updateFromModel(this.currentClassifier);

    if (stepData.supportVectorIndices) {
      this.boundaryMesh.renderSupportVectors(this.trainingData.X, stepData.supportVectorIndices);
    } else if (stepData.splits) {
      this.boundaryMesh.renderTreeSplits(stepData.splits);
    }

    // Update metrics and chart
    if (stepData.accuracy !== undefined) {
      this.hud.update({ accuracy: stepData.accuracy });
      this.charts.updateMetrics(stepData.accuracy, stepData.loss, stepData.step);
    }

    // Compute confusion matrix
    const preds = this.trainingData.X.map(x => this.currentClassifier.predict(x));
    const metrics = calculateMetrics(this.trainingData.y, preds);
    this.charts.updateConfusionMatrix(metrics.confusionMatrix);

    return true;
  }

  resetTraining() {
    this.isTrainingPlaying = false;
    this.bottomControls.setPlaying(false);
    this.initClassifier();
    this.charts.reset();
    this.boundaryMesh.clearOverlay();
    this.boundaryMesh.updateFromModel(this.currentClassifier);
  }

  changeDataset(datasetType) {
    this.params.dataset = datasetType;
    this.spawner.loadDataset(datasetType, { noise: this.params.noise });
    this.spawner.clearAll();
    this.resetTraining();
    sounds.playClick();
  }

  changeClassifier(classifierId) {
    this.params.classifier = classifierId;
    this.initClassifier();
    this.resetTraining();
    sounds.playClick();
  }

  handleAction(action) {
    switch (action) {
      case 'freeze':
        this.setMode(this.params.mode === 'FREEZE' ? 'FLOW' : 'FREEZE');
        break;
      case 'train':
        this.setMode('TRAIN');
        break;
      case 'test':
        this.setMode('TEST');
        break;
      case 'challenge':
        this.setMode('CHALLENGE');
        break;
      case 'orbit': {
        const isOrbit = this.toggleOrbit();
        break;
      }
      case 'reset':
        this.scoring.reset();
        this.hud.update({ score: 0, lives: 3, combo: 0, accuracy: null, playerAccuracy: null });
        this.spawner.clearAll();
        this.resetTraining();
        this.setMode('FLOW');
        sounds.playClick();
        break;
    }
  }

  bindEvents() {
    const dom = this.container;

    // Mouse / Touch Slicing
    dom.addEventListener('pointerdown', (e) => {
      if (this.controls.enabled) return;
      this.slicing.startSlash(e.clientX, e.clientY);
      this.bladeTrail.addPoint(this.slicing.currentWorldPos);
    });

    dom.addEventListener('pointermove', (e) => {
      if (this.controls.enabled) return;
      this.slicing.moveSlash(e.clientX, e.clientY);
      if (this.slicing.isPointerDown) {
        this.bladeTrail.addPoint(this.slicing.currentWorldPos);
      }

      // Tooltip inspect on hover
      this.checkHoverTooltip(e.clientX, e.clientY);
    });

    window.addEventListener('pointerup', () => {
      this.slicing.endSlash();
      // If player drew a boundary slash, evaluate it!
      if (this.slicing.lastSlashBoundary) {
        const { w, b } = this.slicing.lastSlashBoundary;
        const targetPts = this.spawner.activeObjects.length > 0
          ? this.spawner.activeObjects
          : this.trainingData.X.map((pt, i) => ({ position: { x: pt[0], y: pt[1] }, userData: { label: this.trainingData.y[i] } }));

        const playerAcc = this.scoring.evaluatePlayerLine(w, b, targetPts);
        this.hud.update({ playerAccuracy: playerAcc });
        sounds.playClick();
      }
    });

    // Sound toggle button in HUD
    this.hud.soundBtn.addEventListener('click', () => {
      const isMuted = sounds.toggleMute();
      this.hud.soundBtn.textContent = isMuted ? '🔇' : '🔊';
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      const key = e.key.toUpperCase();

      if (key === ' ') {
        e.preventDefault();
        this.handleAction('freeze');
      } else if (key === 'T') {
        this.handleAction('train');
      } else if (key === 'R') {
        this.handleAction('reset');
      } else if (key === 'O') {
        this.handleAction('orbit');
      } else if (key === 'D') {
        // Cycle dataset
        const datasets = ['separable', 'overlapping', 'xor', 'moons', 'circles', 'spirals'];
        const nextIdx = (datasets.indexOf(this.params.dataset) + 1) % datasets.length;
        this.changeDataset(datasets[nextIdx]);
      } else if (key >= '1' && key <= '9') {
        const keys = Object.keys(CLASSIFIERS);
        const idx = parseInt(key, 10) - 1;
        if (keys[idx]) {
          this.changeClassifier(keys[idx]);
        }
      }
    });
  }

  checkHoverTooltip(clientX, clientY) {
    const worldPos = this.slicing.getPointerWorldPos(clientX, clientY);
    let hovered = null;

    for (const obj of this.spawner.activeObjects) {
      if (obj.position.distanceTo(worldPos) < 0.6) {
        hovered = obj;
        break;
      }
    }

    if (hovered) {
      let predictedLabel, confidence;
      if (this.currentClassifier && this.currentClassifier.isFitted) {
        predictedLabel = this.currentClassifier.predict([hovered.position.x, hovered.position.y]);
        confidence = this.currentClassifier.predictProba([hovered.position.x, hovered.position.y]);
        if (predictedLabel < 0) confidence = 1.0 - confidence;
      }
      this.tooltips.show(clientX, clientY, {
        type: hovered.userData.type,
        x: hovered.position.x,
        y: hovered.position.y,
        predictedLabel,
        confidence
      });
    } else {
      this.tooltips.hide();
    }
  }

  animate(currentTime) {
    requestAnimationFrame(this.animate);

    const dt = Math.min(0.05, (currentTime - this.lastTime) / 1000);
    this.lastTime = currentTime;

    const isFrozen = this.params.mode === 'FREEZE' || this.params.mode === 'TRAIN';

    // Training step loop
    if (this.isTrainingPlaying) {
      const interval = 1000 / this.trainingSpeed;
      if (currentTime - this.lastTrainingStepTime >= interval) {
        this.lastTrainingStepTime = currentTime;
        this.stepTraining();
      }
    }

    // Slicing logic
    if (this.slicing.isPointerDown && !this.controls.enabled) {
      const slicedHits = this.slicing.sliceCheck(this.spawner.activeObjects);
      for (const hit of slicedHits) {
        const res = this.spawner.sliceObject(hit.target, hit.cutNormal);
        if (res) {
          if (res.isFruit) {
            const { points, combo } = this.scoring.onFruitSliced();
            this.hud.update({ score: this.scoring.score, combo });
          } else {
            const { lives, isGameOver } = this.scoring.onBombHit();
            this.hud.update({ lives, combo: 0 });
          }
        }
      }
    }

    // Auto-slice check in TEST mode
    if (this.params.mode === 'TEST') {
      for (const obj of this.spawner.activeObjects) {
        if (obj.userData.autoSliced && !obj.userData.isSliced) {
          const res = this.spawner.sliceObject(obj, new THREE.Vector3(1, 0, 0));
          if (res) {
            if (res.isFruit) {
              const { points, combo } = this.scoring.onFruitSliced();
              this.hud.update({ score: this.scoring.score, combo });
            } else {
              // Model misclassified a bomb as a fruit and sliced it!
              const { lives } = this.scoring.onBombHit();
              this.hud.update({ lives, combo: 0 });
            }
          }
        }
      }
    }

    // Update systems
    this.spawner.update(dt, isFrozen, this.params.mode, this.currentClassifier);

    for (const obj of this.spawner.activeObjects) {
      this.physics.updateObject(obj, dt);
    }
    for (const half of this.spawner.activeHalves) {
      this.physics.updateHalf(half, dt);
    }

    this.particles.update(dt);
    this.bladeTrail.update();
    this.scoring.update(dt);

    if (this.controls.enabled) {
      this.controls.update();
    }

    // Render with bloom postprocessing
    this.postProcessing.composer.render();
  }
}

// Start application
window.addEventListener('DOMContentLoaded', () => {
  new MLNinjaApp();
});

