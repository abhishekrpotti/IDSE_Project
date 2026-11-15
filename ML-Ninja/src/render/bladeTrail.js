import * as THREE from 'three';

export class BladeTrail {
  constructor(scene, maxPoints = 24) {
    this.scene = scene;
    this.maxPoints = maxPoints;
    this.points = []; // [{ pos: Vector3, time: number }]
    this.halfWidth = 0.12;

    // BufferGeometry for dynamic ribbon
    const numVertices = this.maxPoints * 2;
    this.geometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(numVertices * 3);
    this.colors = new Float32Array(numVertices * 4); // RGBA

    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('aColor', new THREE.BufferAttribute(this.colors, 4));

    // Custom shader material for glowing blade
    this.material = new THREE.ShaderMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      vertexShader: `
        attribute vec4 aColor;
        varying vec4 vColor;
        void main() {
          vColor = aColor;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec4 vColor;
        void main() {
          gl_FragColor = vColor;
        }
      `
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
  }

  addPoint(pos) {
    this.points.unshift({ pos: pos.clone(), time: performance.now() });
    if (this.points.length > this.maxPoints) {
      this.points.pop();
    }
  }

  clear() {
    this.points = [];
    this.update();
  }

  update() {
    const now = performance.now();
    // Remove expired points older than 180ms
    this.points = this.points.filter(p => now - p.time < 180);

    const count = this.points.length;
    if (count < 2) {
      // Clear vertices
      this.positions.fill(0);
      this.colors.fill(0);
      this.geometry.attributes.position.needsUpdate = true;
      this.geometry.attributes.aColor.needsUpdate = true;
      return;
    }

    let vertexIdx = 0;
    let colorIdx = 0;

    for (let i = 0; i < count; i++) {
      const p = this.points[i].pos;
      const t = i / (count - 1); // 0 at head (newest), 1 at tail (oldest)
      const width = this.halfWidth * (1.0 - t * 0.85);
      const alpha = Math.max(0, (1.0 - t) * 0.9);

      // Compute perpendicular 2D direction in XY plane
      let dir = new THREE.Vector3(0, 1, 0);
      if (i < count - 1) {
        dir.subVectors(this.points[i + 1].pos, p).normalize();
      } else if (i > 0) {
        dir.subVectors(p, this.points[i - 1].pos).normalize();
      }
      const normal = new THREE.Vector3(-dir.y, dir.x, 0).multiplyScalar(width);

      // Left vertex
      this.positions[vertexIdx++] = p.x + normal.x;
      this.positions[vertexIdx++] = p.y + normal.y;
      this.positions[vertexIdx++] = p.z + 0.05;

      // Electric cyan / neon blade color
      this.colors[colorIdx++] = 0.2;
      this.colors[colorIdx++] = 0.9;
      this.colors[colorIdx++] = 1.0;
      this.colors[colorIdx++] = alpha;

      // Right vertex
      this.positions[vertexIdx++] = p.x - normal.x;
      this.positions[vertexIdx++] = p.y - normal.y;
      this.positions[vertexIdx++] = p.z + 0.05;

      this.colors[colorIdx++] = 0.1;
      this.colors[colorIdx++] = 0.7;
      this.colors[colorIdx++] = 1.0;
      this.colors[colorIdx++] = alpha;
    }

    // Zero out remaining buffer
    const totalVertices = this.maxPoints * 2;
    while (vertexIdx < totalVertices * 3) {
      this.positions[vertexIdx++] = 0;
    }
    while (colorIdx < totalVertices * 4) {
      this.colors[colorIdx++] = 0;
    }

    this.geometry.setDrawRange(0, count * 2);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.aColor.needsUpdate = true;
  }
}
