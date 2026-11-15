import * as THREE from 'three';

export class ParticleSystem {
  constructor(scene, maxParticles = 600) {
    this.scene = scene;
    this.maxParticles = maxParticles;
    this.particles = [];

    const geom = new THREE.BufferGeometry();
    this.positions = new Float32Array(maxParticles * 3);
    this.colors = new Float32Array(maxParticles * 3);
    this.sizes = new Float32Array(maxParticles);

    geom.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    geom.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1));

    this.material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
      uniforms: {},
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          float d = length(gl_PointCoord - vec2(0.5));
          if (d > 0.5) discard;
          float alpha = smoothstep(0.5, 0.05, d);
          gl_FragColor = vec4(vColor, alpha);
        }
      `
    });

    this.points = new THREE.Points(geom, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
  }

  emitJuice(pos, hexColor = 0x00e676, count = 25) {
    const col = new THREE.Color(hexColor);
    for (let i = 0; i < count; i++) {
      if (this.particles.length >= this.maxParticles) {
        this.particles.shift();
      }
      const angle = Math.random() * Math.PI * 2;
      const speed = 1.5 + Math.random() * 3.5;
      this.particles.push({
        x: pos.x,
        y: pos.y,
        z: pos.z,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed + 1.2,
        vz: (Math.random() - 0.5) * 2.0,
        r: col.r,
        g: col.g,
        b: col.b,
        size: 0.18 + Math.random() * 0.15,
        life: 1.0,
        decay: 1.8 + Math.random() * 1.5
      });
    }
  }

  emitExplosion(pos, count = 50) {
    for (let i = 0; i < count; i++) {
      if (this.particles.length >= this.maxParticles) {
        this.particles.shift();
      }
      const angle = Math.random() * Math.PI * 2;
      const speed = 2.0 + Math.random() * 5.0;
      const isFire = Math.random() > 0.4;
      this.particles.push({
        x: pos.x,
        y: pos.y,
        z: pos.z,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed + 1.5,
        vz: (Math.random() - 0.5) * 3.0,
        r: isFire ? 1.0 : 0.4,
        g: isFire ? 0.3 + Math.random() * 0.4 : 0.4,
        b: 0.1,
        size: 0.25 + Math.random() * 0.2,
        life: 1.0,
        decay: 1.2 + Math.random() * 1.0
      });
    }
  }

  update(dt) {
    const gravity = -9.8;
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.life -= p.decay * dt;
      if (p.life <= 0) {
        this.particles.splice(i, 1);
        continue;
      }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
      p.vy += gravity * dt;
    }

    const count = this.particles.length;
    for (let i = 0; i < count; i++) {
      const p = this.particles[i];
      this.positions[i * 3 + 0] = p.x;
      this.positions[i * 3 + 1] = p.y;
      this.positions[i * 3 + 2] = p.z;

      this.colors[i * 3 + 0] = p.r * p.life;
      this.colors[i * 3 + 1] = p.g * p.life;
      this.colors[i * 3 + 2] = p.b * p.life;

      this.sizes[i] = p.size * p.life;
    }

    // Zero out unused
    for (let i = count; i < this.maxParticles; i++) {
      this.positions[i * 3 + 0] = 0;
      this.positions[i * 3 + 1] = 0;
      this.positions[i * 3 + 2] = 0;
      this.sizes[i] = 0;
    }

    this.points.geometry.attributes.position.needsUpdate = true;
    this.points.geometry.attributes.color.needsUpdate = true;
    this.points.geometry.attributes.size.needsUpdate = true;
  }
}

