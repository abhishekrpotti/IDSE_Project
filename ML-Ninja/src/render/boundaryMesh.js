import * as THREE from 'three';

export class BoundaryMesh {
  constructor(scene, options = {}) {
    this.scene = scene;
    this.resolution = options.resolution ?? 128;
    this.bounds = {
      xMin: -4.5,
      xMax: 4.5,
      yMin: -3.0,
      yMax: 4.5
    };

    // 128x128 RGBA DataTexture
    const size = this.resolution * this.resolution;
    this.textureData = new Uint8Array(4 * size);
    // Initialize with neutral dark blue/purple
    for (let i = 0; i < size; i++) {
      this.textureData[i * 4 + 0] = 18;  // R
      this.textureData[i * 4 + 1] = 22;  // G
      this.textureData[i * 4 + 2] = 34;  // B
      this.textureData[i * 4 + 3] = 160; // Alpha
    }

    this.texture = new THREE.DataTexture(
      this.textureData,
      this.resolution,
      this.resolution,
      THREE.RGBAFormat
    );
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.needsUpdate = true;

    // Plane geometry matching the feature coordinate bounds
    const width = this.bounds.xMax - this.bounds.xMin;
    const height = this.bounds.yMax - this.bounds.yMin;
    const centerX = (this.bounds.xMax + this.bounds.xMin) / 2;
    const centerY = (this.bounds.yMax + this.bounds.yMin) / 2;

    this.geometry = new THREE.PlaneGeometry(width, height);

    // Custom shader material for smooth contour line at 0.5
    this.material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uTexture: { value: this.texture },
        uOpacity: { value: 0.55 },
        uShowGrid: { value: 1.0 }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uTexture;
        uniform float uOpacity;
        uniform float uShowGrid;
        varying vec2 vUv;

        void main() {
          vec4 col = texture2D(uTexture, vUv);
          
          // Coordinate grid lines
          vec2 grid = abs(fract(vUv * 10.0 - 0.5) - 0.5) / fwidth(vUv * 10.0);
          float line = min(grid.x, grid.y);
          float gridAlpha = (1.0 - min(line, 1.0)) * 0.15 * uShowGrid;

          // Main axis lines at center
          vec2 axis = abs(vUv - 0.5) / fwidth(vUv);
          float axisLine = min(axis.x, axis.y);
          float axisAlpha = (1.0 - min(axisLine, 1.0)) * 0.35 * uShowGrid;

          vec3 finalColor = mix(col.rgb, vec3(0.5, 0.7, 1.0), gridAlpha + axisAlpha);
          float finalAlpha = max(col.a * uOpacity, (gridAlpha + axisAlpha) * 0.8);

          gl_FragColor = vec4(finalColor, finalAlpha);
        }
      `
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.position.set(centerX, centerY, -0.05);
    this.mesh.receiveShadow = false;
    this.scene.add(this.mesh);

    // Group for vector overlays: support vector rings, margin lines, tree splits
    this.overlayGroup = new THREE.Group();
    this.overlayGroup.position.z = 0.02;
    this.scene.add(this.overlayGroup);
  }

  updateFromModel(classifier) {
    if (!classifier) return;

    const res = this.resolution;
    const { xMin, xMax, yMin, yMax } = this.bounds;

    for (let j = 0; j < res; j++) {
      const y = yMin + (j / (res - 1)) * (yMax - yMin);
      for (let i = 0; i < res; i++) {
        const x = xMin + (i / (res - 1)) * (xMax - xMin);
        const prob = classifier.predictProba([x, y]); // in [0, 1]

        const idx = (j * res + i) * 4;

        // Contour line near prob = 0.5 (within ± 0.03)
        const isContour = Math.abs(prob - 0.5) < 0.025;

        if (isContour) {
          // Bright white-gold contour line
          this.textureData[idx + 0] = 255;
          this.textureData[idx + 1] = 240;
          this.textureData[idx + 2] = 200;
          this.textureData[idx + 3] = 230;
        } else if (prob > 0.5) {
          // Green side (Fruit class +1)
          const intensity = Math.min(1.0, (prob - 0.5) * 2.0);
          this.textureData[idx + 0] = Math.round(15 + 20 * intensity);
          this.textureData[idx + 1] = Math.round(120 + 90 * intensity);
          this.textureData[idx + 2] = Math.round(40 + 30 * intensity);
          this.textureData[idx + 3] = Math.round(100 + 70 * intensity);
        } else {
          // Red side (Bomb class -1)
          const intensity = Math.min(1.0, (0.5 - prob) * 2.0);
          this.textureData[idx + 0] = Math.round(140 + 95 * intensity);
          this.textureData[idx + 1] = Math.round(20 + 20 * intensity);
          this.textureData[idx + 2] = Math.round(30 + 30 * intensity);
          this.textureData[idx + 3] = Math.round(100 + 70 * intensity);
        }
      }
    }

    this.texture.needsUpdate = true;
  }

  clearOverlay() {
    while (this.overlayGroup.children.length > 0) {
      const child = this.overlayGroup.children.pop();
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
  }

  renderSupportVectors(points, indices) {
    this.clearOverlay();
    const ringGeom = new THREE.RingGeometry(0.55, 0.65, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffd600,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85
    });

    for (const idx of indices) {
      const pt = points[idx];
      if (!pt) continue;
      const mesh = new THREE.Mesh(ringGeom, ringMat);
      mesh.position.set(pt[0], pt[1], 0.05);
      this.overlayGroup.add(mesh);
    }
  }

  renderNeighborLines(queryPt, neighbors) {
    this.clearOverlay();
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.75,
      linewidth: 2
    });

    for (const n of neighbors) {
      const points = [
        new THREE.Vector3(queryPt[0], queryPt[1], 0.05),
        new THREE.Vector3(n.point[0], n.point[1], 0.05)
      ];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geom, lineMat);
      this.overlayGroup.add(line);
    }
  }

  renderTreeSplits(splits) {
    this.clearOverlay();
    const splitMat = new THREE.LineBasicMaterial({
      color: 0x00e676,
      transparent: true,
      opacity: 0.8,
      linewidth: 2
    });

    for (const split of splits) {
      let p1, p2;
      if (split.feature === 0) {
        // Vertical split line at x = threshold
        p1 = new THREE.Vector3(split.threshold, split.bounds.yMin, 0.04);
        p2 = new THREE.Vector3(split.threshold, split.bounds.yMax, 0.04);
      } else {
        // Horizontal split line at y = threshold
        p1 = new THREE.Vector3(split.bounds.xMin, split.threshold, 0.04);
        p2 = new THREE.Vector3(split.bounds.xMax, split.threshold, 0.04);
      }
      const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
      const line = new THREE.Line(geom, splitMat);
      this.overlayGroup.add(line);
    }
  }

  setVisible(visible) {
    this.mesh.visible = visible;
    this.overlayGroup.visible = visible;
  }
}

