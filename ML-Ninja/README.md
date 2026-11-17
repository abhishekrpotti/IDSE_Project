# ML Ninja 🍉

[Click here to play the game online]https://arpitsrivastavain.github.io/ML-Ninja/

An interactive, static browser game about classical machine learning. Every fruit and bomb is a labeled point; draw the decision boundary that separates the two classes.

## Run locally

```sh
npm install
npm run dev
```

## Build and preview

```sh
npm run build
npx serve dist
```

Vite uses relative asset paths, so the build works from a subfolder and needs no backend, API key, CDN, or remote assets.

## Deploy

- **GitHub Pages:** push to `main` with Pages configured to use GitHub Actions. The included `.github/workflows/deploy.yml` builds and deploys `dist`.
- **Netlify:** run `npm run build`, then drag the `dist` folder into Netlify Drop.
- **Vercel:** import the repository; use `npm run build` as the build command and `dist` as the output directory.

## How to play

Press **Space** or Start wave. Watch the data rise, then draw a boundary during the 10-second slow-motion window. Perceptron, Logistic Regression, and Linear SVM use a line; the other classifiers accept a freehand boundary. Swiping slices objects while they launch; during the drawing window, switch to Slice mode to slice. Press Enter or Submit to finish early. Each fruit hit adds 10 points, each bomb hit subtracts 8 points, and boundary agreement can add up to 100 points. On the results screen, use **1–9** to choose a classifier, **D** to cycle datasets, or **R** to retry.

The models are implemented in plain JavaScript without an ML dependency. Dataset generation is seeded for reproducible examples. Graphics use a small Three.js scene and HTML overlays; sound is synthesized with WebAudio.

## What the game teaches

ML Ninja makes the classification problem visible. Each item has two numeric features `(x1, x2)` and a known label: fruit is `+1`, bomb is `-1`. Each classifier trains on 100 generated examples, then predicts a 2D region and labels the current wave. During the round, the player slices items and draws a line or freehand boundary. The results card compares the player's classifications and the model's predictions against the wave's true labels. It shows accuracy, a confusion matrix, fruit precision/recall/F1, score, and a decision-region plot. The score combines fruit and bomb hits with a bonus based on how closely the player's boundary agrees with the model. Model success depends on whether the pattern fits its assumptions; low scores on a particular dataset are part of the lesson, not a promise that every classifier can solve every wave.

The left panel controls wave size (8–24 points), feature noise (0–0.5), label flips (0–50%), and the selected model's supported hyperparameters. Training size is fixed at 100 examples. The generators use the seeded RNG in `src/data/rng.js`; reusing a seed and settings reproduces the same points. Label flips are applied after the geometric class is generated. Outlier blobs additionally put a small fraction of points on the opposite side of their original class boundary.

## Classifiers

| Classifier | Algorithm | Boundary | Available controls | Strengths and limits |
|---|---|---|---|---|
| Perceptron | Seed-shuffled mistake-driven weight updates | Straight line | learning rate | Simple, quick linear baseline; can wobble with overlap and outliers. |
| Logistic Regression | Gradient descent on sigmoid log-loss with L2 | Straight line; sigmoid probabilities | learning rate | Estimates a fruit probability as well as a class; cannot fit nonlinear shapes. |
| Linear SVM | L2 plus hinge-loss subgradient updates | Maximum-margin line | C, learning rate | Focuses on margin violations; still linear. Result graphs mark the separator, margins, and support vectors. |
| Kernel SVM | RBF kernel with deterministic simplified SMO | Smooth nonlinear | C, gamma | Wraps around curved clusters; high gamma can overfit and small training samples can be unstable. Result graphs mark its boundary, margin contours, and support vectors. |
| k-NN | Euclidean vote among standardized neighbors | Local and jagged | k | Flexible and intuitive; small k follows noise, large k smooths details. |
| Gaussian Naive Bayes | Per-class diagonal Gaussian likelihoods and priors | Axis-aligned quadratic | variance floor | Fast probabilistic baseline; ignores feature correlation. Results explain priors, likelihoods, means, variances, and the independence assumption. |
| LDA | Shared pooled full covariance and class priors | Straight line | ridge | Uses correlated feature structure; assumes the classes share covariance. |
| QDA | Separate full covariance and priors by class | Quadratic | ridge | Fits unequal class shapes; needs enough examples for stable covariance estimates. |
| Decision Tree | Greedy Gini splits | Axis-aligned boxes | max depth | Learns checkerboard-like rules; deep trees can memorize noise. |
| Random Forest | Bootstrap trees with random feature subsets at each split | Averaged blocky regions | trees, max depth | Averaging reduces a single tree's variance; boundaries remain blocky and small forests can miss local patterns. |

The models expose `fit(X, y, params)`, `predict(x)`, `predictProba(x)` where applicable, and metadata from `src/classifiers/index.js`. Training is deterministic for a fixed seed and does not use an external ML library.

## Datasets

| Dataset | How points are made | What it demonstrates |
|---|---|---|
| Separable blobs | Two Gaussian clouds with separated means | Linear methods can solve a clean split. |
| Overlapping blobs | Two closer, noisier Gaussian clouds | Bayes error and the effect of noise/regularization. |
| XOR | Alternating diagonal quadrants | A single line cannot separate the labels. |
| Two moons | Two interleaved noisy half circles | Curved nonlinear regions. |
| Circles | Inner disk and outer ring | Closed radial boundary; Naive Bayes can also fit concentric classes because the variances differ by axis. |
| Spirals | Two noisy arms with opposite phase | A highly turning boundary; many defaults struggle, especially shallow trees. |
| Correlated blobs | Same tilted Gaussian covariance, different means | LDA's pooled full covariance can outperform diagonal Naive Bayes. |
| Unequal-variance blobs | Tight class beside a diffuse class | QDA can fit different covariance matrices; model performance varies with sample seed. |
| Outlier blobs | Separated clouds with a few opposite-side mislabeled points | Perceptron and tiny-k neighbors are sensitive; regularization helps. |
| Checkerboard (3x3) | Alternating labels in axis-aligned squares | Trees and forests fit boxy boundaries; linear models are weak. |

The noise slider adjusts Gaussian spread for each generator. The label-flip slider randomly reverses labels after point generation. Wave points and model training points are generated separately from the same selected dataset and settings, so accuracy measures generalization to the current wave.

## Validation and benchmark

```sh
npm test
npm run bench
npm run build
```

`npm run bench` evaluates every classifier on every dataset over 20 seeds and prints mean accuracy ± population standard deviation. The benchmark uses 120 training and 180 test points per seed. The test suite also checks seeded data, nonlinear-vs-linear expectations, covariance assumptions, forest stability, classifier distinctness, game round statistics, and slider bounds.

### 20-seed benchmark (% mean ± standard deviation)

Measured with the current defaults, noise 0.2, and no label flips:

| Classifier | Separable | Overlap | XOR | Moons | Circles | Spirals | Correlated | Unequal variance | Outliers | Checkerboard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Perceptron | 99.6 ± 0.5 | 72.6 ± 6.4 | 56.2 ± 8.5 | 83.7 ± 4.3 | 51.4 ± 8.4 | 53.3 ± 7.5 | 96.3 ± 2.3 | 67.9 ± 13.9 | 87.1 ± 8.5 | 53.4 ± 5.5 |
| Logistic Regression | 99.2 ± 0.6 | 79.1 ± 2.7 | 51.5 ± 5.0 | 86.4 ± 2.1 | 51.6 ± 3.5 | 62.7 ± 4.0 | 86.3 ± 5.9 | 80.5 ± 3.1 | 91.0 ± 1.9 | 53.3 ± 5.3 |
| Linear SVM | 99.4 ± 0.4 | 78.9 ± 2.7 | 51.5 ± 4.9 | 85.3 ± 2.4 | 51.6 ± 3.6 | 62.4 ± 3.8 | 69.3 ± 5.7 | 75.1 ± 3.3 | 91.2 ± 2.0 | 53.8 ± 4.5 |
| Kernel SVM | 99.4 ± 0.7 | 71.4 ± 7.2 | 99.8 ± 0.4 | 99.7 ± 0.6 | 100.0 ± 0.0 | 94.5 ± 2.5 | 92.1 ± 5.9 | 90.8 ± 3.6 | 90.1 ± 4.5 | 91.1 ± 6.4 |
| k-NN | 99.7 ± 0.3 | 76.7 ± 3.0 | 100.0 ± 0.1 | 99.7 ± 0.4 | 99.7 ± 1.0 | 89.8 ± 3.4 | 94.6 ± 2.1 | 90.1 ± 2.7 | 91.6 ± 2.0 | 95.0 ± 3.2 |
| Gaussian Naive Bayes | 99.8 ± 0.3 | 79.0 ± 2.8 | 50.7 ± 5.6 | 87.2 ± 2.1 | 99.9 ± 0.3 | 61.2 ± 3.8 | 67.8 ± 5.1 | 92.4 ± 1.9 | 91.0 ± 2.0 | 56.8 ± 6.6 |
| LDA | 99.9 ± 0.2 | 79.0 ± 2.7 | 51.5 ± 5.0 | 87.2 ± 2.2 | 51.7 ± 3.5 | 62.8 ± 4.1 | 97.4 ± 1.4 | 80.1 ± 2.7 | 90.7 ± 1.8 | 53.2 ± 5.3 |
| QDA | 99.8 ± 0.3 | 78.7 ± 2.9 | 99.9 ± 0.2 | 86.7 ± 2.3 | 99.8 ± 0.8 | 60.8 ± 4.0 | 95.3 ± 2.6 | 89.4 ± 1.7 | 90.9 ± 1.8 | 56.5 ± 6.0 |
| Decision Tree | 99.8 ± 0.4 | 71.2 ± 2.9 | 98.5 ± 1.9 | 96.8 ± 2.5 | 98.8 ± 0.9 | 84.9 ± 7.7 | 88.1 ± 2.7 | 89.9 ± 2.8 | 91.9 ± 1.8 | 91.1 ± 9.5 |
| Random Forest | 99.7 ± 0.4 | 75.6 ± 2.4 | 97.9 ± 2.0 | 96.4 ± 1.7 | 99.0 ± 1.1 | 86.6 ± 3.8 | 89.3 ± 3.7 | 91.6 ± 2.8 | 91.9 ± 1.9 | 82.1 ± 6.0 |

Accuracy is data- and seed-dependent. In particular, the benchmark shows that some classifiers do not match their idealized dataset: the default tree/forest are below the best local models on checkerboard, and most models are weaker on spirals. These are useful demonstrations of model assumptions and hyperparameter limits.


## Learning and accessibility features

- Model-specific controls and a live decision-region preview update from the selected dataset and settings.
- Compare-all displays ten small decision-region graphs scored on the same wave, marking the best and lowest performers. Every graph uses the same fruit-circle and bomb-diamond markers; linear and kernel SVM graphs also show the model separator and margin lines.
- Results include the model's boundary shape, a data-dependent explanation, precision/recall/F1, a beat-the-model badge, and the model's decision regions. SVM result graphs additionally highlight support vectors and margins; logistic regression shows its sigmoid with the wave points, and k-NN and Gaussian Naive Bayes include learning notes.
- A first-visit Quick Guide walks through the lab, datasets, display and wave settings, play area, help panel, and result card. It also gives a project checklist covering question framing, model comparison, error interpretation, reproducibility, and the limits of this synthetic-data demo. Reopen it any time with the **Guide** button.
- Local progress keeps the best player/model accuracy for each model/dataset pair; the HUD tracks a no-bomb streak.
- Pick-the-model challenges score a guess against the current wave's ten-model comparison. Mystery-model challenges hide the chosen model until results and ask for its family. Four guided lessons demonstrate linear/XOR, LDA/Naive Bayes, tree/forest, and k-neighbor contrasts.
- Colorblind mode changes the plot palette and uses different point shapes for the two classes. Reduced motion disables the result-card entrance animation and shortens CSS transitions; controls remain keyboard-operable.

The comparison screen computes each classifier on demand from the active training sample. The current-wave Pick-the-model reveal ranks those fits on the wave rather than using a stored benchmark lookup; the full 20-seed benchmark is available through `npm run bench` and in the table above. Local scoreboard values stay in this browser's localStorage.
