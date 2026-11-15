# 🥷 ML Ninja — Fruit Ninja × Classical Machine Learning

An interactive 3D Fruit Ninja-style educational game that intuitively demonstrates how classical machine learning algorithms discover decision boundaries in feature space. Built with vanilla JavaScript (ES modules), Vite, and Three.js — with zero external ML dependencies!

---

## 🎮 Core Concept & Game Mechanics

- **Feature Space**: Every flying object launched into the arena represents a 2D feature data point $(x_1, x_2)$ (or optionally 3D in inspect mode).
- **Classes**:
  - **Fruits (Class +1)**: Watermelon, Orange, Apple, Kiwi, Pineapple. Slicing gives points and increments combo multipliers!
  - **Bombs (Class -1)**: Black iron spheres with burning fuses and pulsing hazard LEDs. Slicing detonates an explosion and deducts a life.
- **Datasets**: Toggle between Linearly Separable Blobs, Overlapping Blobs, XOR, Two Moons, Concentric Circles, and Spirals.
- **Blade Trail**: Glowing neon ribbon following the cursor with additive blending and bloom postprocessing.
- **Player Slash as Decision Boundary**: Slicing across frozen points evaluates your hand-drawn linear separation line $w_1 x_1 + w_2 x_2 + b = 0$ against the true labels in real time!
- **Auto-Pilot Test Phase**: The trained algorithm predicts flying objects in real time. Predicted fruits receive glowing green holographic HUD rings and are auto-sliced; predicted bombs receive red hazard markers. When the model makes a mistake, the player witnesses the error directly (e.g., detonating a bomb false positive or missing a fruit)!

---

## 🕹️ Controls & Shortcuts

| Key / Action | Function |
| :--- | :--- |
| **Mouse Drag / Touch** | Draw glowing blade trail to slice fruits & avoid bombs |
| **SPACE** | Freeze / Unfreeze time (converts flying objects to static labeled scatter plot) |
| **T** | Train the active machine learning classifier step-by-step |
| **R** | Reset score, lives, and training state |
| **O** | Toggle 3D OrbitControls (inspect feature depth) |
| **D** | Cycle through synthetic datasets (Separable, XOR, Moons, Circles, Spirals) |
| **1 – 9** | Instant switch between ML algorithms |

---

## 🧠 Classifiers Implemented from Scratch

All 10 algorithms are implemented natively in vanilla ES modules with step-by-step generator functions (`*fitSteps`) for live visual training:

1. **Perceptron**: Classical linear threshold unit. Animates line rotations and shifts upon encountering misclassified points.
2. **Logistic Regression**: Gradient descent on binary cross-entropy loss with live loss curve and probability sigmoid heatmaps.
3. **Linear SVM**: Hinge loss subgradient descent displaying max-margin gutters ($w \cdot x + b = \pm 1$) and highlighted support vectors.
4. **Kernel SVM (RBF & Poly)**: Non-linear kernel projection solving circular and interleaved distributions with adjustable $C$ and $\gamma$.
5. **k-Nearest Neighbors (k-NN)**: Instance-based majority vote with real-time distance neighbor lines and $L_1 / L_2$ metrics.
6. **Gaussian Naive Bayes**: Generative model estimating class priors, feature means, and variance ellipses under conditional independence.
7. **LDA & QDA**: Linear & Quadratic Discriminant Analysis comparing pooled vs class-specific covariance structures.
8. **Decision Tree (CART)**: Greedy orthogonal axis-aligned partition lines optimizing Gini impurity or Shannon entropy.
9. **Random Forest**: Ensemble of bagged decision trees with out-of-bag diversity and consensus voting.
10. **Multi-Layer Perceptron (MLP)**: 1-hidden-layer neural network with Tanh and Sigmoid activations trained via backpropagation.

---

## 🚀 Getting Started

```bash
# Install dependencies
npm install

# Run unit tests (Vitest)
npm run test

# Launch local dev server
npm run dev
```

