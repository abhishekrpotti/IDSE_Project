# ML Ninja — Implementation Plan

## Objective
Build a lightweight, static Fruit Ninja-style learning game that teaches classifier boundaries through play. Use vanilla JavaScript ES modules, Vite, three.js, and GSAP, with no remote assets, APIs, or runtime services.

## Delivery approach
Implement in the seven requested milestones. Keep each milestone independently runnable, and verify the browser experience at its completion with screenshots and console checks. Keep the graphics inexpensive: shared low-poly meshes and materials, lightweight canvas textures, a pooled particle system, no real-time shadows or postprocessing, and a capped pixel ratio. Build and run the production output from a static server before final delivery.

## Milestones

1. **Scene, theme, and wave sizing** — Establish Vite project structure, theme tokens, responsive HTML shell, Three.js scene, toon-like pastel fruit and bomb objects with faces and fake ground blobs, and a bounded wave object pool. Confirm object scale, clear silhouettes, and spacing.
2. **Synchronized launches** — Add seeded dataset samples and hand-written projectile trajectories. Solve launch velocities to put each point at its encoded apex at a shared target time; stagger launch times with compensation. Confirm all objects form the intended scatter without overlap.
3. **Slow-motion draw phase** — Add the IDLE → LAUNCH → DRAW state path, apex transition, slow motion, real-time countdown, dim overlay, drawing input, and state-gated controls. Support mouse and touch.
4. **Evaluation and scoring** — Implement linear-line and nonlinear freehand boundary representations, player/classifier accuracy, verdict thresholds, consequences for fruits and bombs, scoring, and evaluation tests.
5. **Classifiers and boundary visualization** — Implement the requested classical models from scratch behind a shared interface, seeded generators for all datasets, metrics, 64×64 classifier boundary texture, and incremental fit steps. Add focused Vitest coverage for toy data and expected XOR behavior.
6. **Results, picker, and learning panels** — Add the result card, comparison visualization/confusion matrix, classifier and dataset picker, hyperparameter controls, classifier explanations, tutorial, keyboard shortcuts, and responsive layout.
7. **Polish, performance, and hosting** — Add WebAudio effects and mute, pooled slice/explosion particles, quality adaptation and low-quality toggle, idle/hidden-tab render behavior, cleanup on switching, documentation, and GitHub Pages workflow. Verify static output with `npx serve dist` and capture the requested walkthrough screens.

## Technical decisions
- Keep tunables in `/src/config.js` and palette values in `/src/theme.js`, mirrored as CSS custom properties.
- Use simple, shared geometry and material resources; avoid per-object allocations in the render loop. Pool transient effects and cap live particles.
- Keep UI as HTML/CSS overlays and charts as small 2D canvas components. Use WebAudio synthesis instead of audio files.
- Make datasets reproducible with a seeded RNG. Keep classifier training and prediction independent of Three.js and UI code.
- Set Vite `base: './'`; ensure the built app uses only bundled local assets and works from a nested path.

## Verification at each milestone
At each milestone: run the app in a browser, inspect the relevant interaction, capture a screenshot, and check the browser console. Add and run targeted Vitest tests for classifier/evaluation work in milestones 4–5. At the end, run the production build and serve `dist` locally for the final static-build walkthrough.

## Acceptance checklist
- Starts with `npm run dev`; builds to a static `dist` with relative asset paths.
- Pastel cartoon styling clearly distinguishes fruit and bombs; waves contain at most 20 spaced objects and synchronize at the apex.
- Draw phase lasts five real seconds, supports line/freehand modes, and leads to an understandable result with player-versus-model comparison.
- Classifier and dataset changes remain stable and clean up rendering resources.
- Low-end rendering constraints are respected, including adaptive quality, hidden-tab pause, and no expensive rendering features.
- README includes GitHub Pages, Netlify, and Vercel deployment steps; GitHub Actions workflow is included.
- Final walkthrough includes wave launch, draw phase, results, picker, and the production build served statically.

## Approval gate
Implementation will begin after approval of this plan. Browser screenshots will be shared after each milestone as requested.
