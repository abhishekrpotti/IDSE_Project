import * as THREE from 'three';
import { createFruitMesh, createSlicedHalves } from '../render/fruitMeshes.js';
import { createBombMesh } from '../render/bombMesh.js';
import { generateDataset } from '../data/generators.js';

export class Spawner {
  constructor(scene, particles, sounds) {
    this.scene = scene;
    this.particles = particles;
    this.sounds = sounds;

    this.activeObjects = [];
    this.activeHalves = [];
    this.datasetType = 'separable';
    this.dataset = null;
    this.dataIndex = 0;
    this.spawnTimer = 0;
    this.spawnInterval = 1.1; // seconds

    this.fruitSubtypes = ['watermelon', 'orange', 'apple', 'kiwi', 'pineapple'];
    this.loadDataset('separable');
  }

  loadDataset(type = 'separable', options = {}) {
    this.datasetType = type;
    this.dataset = generateDataset(type, { n: 80, noise: 0.15, ...options });
    this.dataIndex = 0;
  }

  spawnSinglePoint(xCoord, yCoord, label) {
    // xCoord, yCoord are feature values in roughly [-4, 4]
    const isFruit = label >= 0;
    let mesh;

    if (isFruit) {
      const subtype = this.fruitSubtypes[Math.floor(Math.random() * this.fruitSubtypes.length)];
      mesh = createFruitMesh(subtype);
    } else {
      mesh = createBombMesh();
    }

    // Launch from bottom edge
    const startX = (Math.random() - 0.5) * 5.0;
    const startY = -4.5;
    mesh.position.set(startX, startY, 0);

    // Ballistic velocity towards feature xCoord
    const launchTime = 1.2 + Math.random() * 0.4;
    const vx = (xCoord - startX) / launchTime;
    // vy chosen so apex reaches around yCoord
    const vy = (yCoord - startY) / launchTime - 0.5 * (-8.5) * launchTime;

    mesh.userData.physics = {
      vx,
      vy,
      vz: (Math.random() - 0.5) * 0.3,
      rotSpeedX: (Math.random() - 0.5) * 4.0,
      rotSpeedY: (Math.random() - 0.5) * 4.0,
      rotSpeedZ: (Math.random() - 0.5) * 4.0,
      isFrozen: false
    };

    mesh.userData.featureX = xCoord;
    mesh.userData.featureY = yCoord;
    mesh.userData.label = label;
    mesh.userData.targetFeaturePos = new THREE.Vector3(xCoord, yCoord, 0);

    // Prediction marker ring (for TEST mode)
    const markerGeom = new THREE.RingGeometry(0.55, 0.65, 24);
    const markerMat = new THREE.MeshBasicMaterial({
      color: 0x00e676,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0
    });
    const marker = new THREE.Mesh(markerGeom, markerMat);
    marker.position.z = 0.05;
    marker.visible = false;
    mesh.add(marker);
    mesh.userData.predictionMarker = marker;

    this.scene.add(mesh);
    this.activeObjects.push(mesh);
    return mesh;
  }

  update(dt, isFrozen = false, mode = 'FLOW', classifier = null) {
    // Spawning logic (only if not frozen)
    if (!isFrozen) {
      this.spawnTimer += dt;
      if (this.spawnTimer >= this.spawnInterval) {
        this.spawnTimer = 0;
        const count = 1 + Math.floor(Math.random() * 2); // 1 or 2 items
        for (let c = 0; c < count; c++) {
          if (this.dataset && this.dataset.X.length > 0) {
            const idx = this.dataIndex % this.dataset.X.length;
            this.dataIndex++;
            const pt = this.dataset.X[idx];
            const lbl = this.dataset.y[idx];
            this.spawnSinglePoint(pt[0], pt[1], lbl);
          }
        }
      }
    }

    // Update active objects
    for (let i = this.activeObjects.length - 1; i >= 0; i--) {
      const obj = this.activeObjects[i];

      if (isFrozen) {
        // Smoothly interpolate towards exact dataset coordinates
        obj.position.lerp(obj.userData.targetFeaturePos, 0.08);
        obj.userData.physics.isFrozen = true;
      } else {
        obj.userData.physics.isFrozen = false;
        // Check if fallen below screen
        if (obj.position.y < -5.5) {
          this.scene.remove(obj);
          this.activeObjects.splice(i, 1);
          continue;
        }
      }

      // If in TEST mode, update prediction marker and auto-slice
      if (mode === 'TEST' && classifier && !obj.userData.isSliced) {
        const prob = classifier.predictProba([obj.position.x, obj.position.y]);
        const pred = classifier.predict([obj.position.x, obj.position.y]);
        const marker = obj.userData.predictionMarker;
        marker.visible = true;
        marker.material.opacity = 0.85;

        if (pred >= 0) {
          marker.material.color.setHex(0x00e676); // Green predicted fruit
        } else {
          marker.material.color.setHex(0xff1744); // Red predicted bomb
        }

        // Auto-slice when reaching apex if model predicts fruit!
        if (pred >= 0 && obj.userData.physics.vy < 0.5 && obj.position.y > -1.5) {
          obj.userData.autoSliced = true;
        }
      }
    }

    // Update sliced halves
    for (let i = this.activeHalves.length - 1; i >= 0; i--) {
      const half = this.activeHalves[i];
      if (half.position.y < -6.0) {
        this.scene.remove(half);
        this.activeHalves.splice(i, 1);
      }
    }
  }

  sliceObject(obj, cutNormal) {
    if (obj.userData.isSliced) return null;
    obj.userData.isSliced = true;

    const pos = obj.position.clone();
    const isFruit = obj.userData.type === 'fruit';

    if (isFruit) {
      this.sounds.playSlice();
      this.sounds.playSplat();

      const { group, half1, half2, innerColor } = createSlicedHalves(obj.userData.fruitType);
      group.position.copy(pos);
      this.scene.add(group);

      // Give halves diverging impulse perpendicular to cut
      const impulse = cutNormal ? cutNormal.clone().multiplyScalar(2.2) : new THREE.Vector3(1.5, 0.5, 0);

      half1.userData.physics = {
        vx: obj.userData.physics.vx + impulse.x,
        vy: obj.userData.physics.vy + impulse.y + 1.2,
        vz: 0,
        rotSpeedX: 4.0,
        rotSpeedY: 3.0
      };

      half2.userData.physics = {
        vx: obj.userData.physics.vx - impulse.x,
        vy: obj.userData.physics.vy - impulse.y + 0.8,
        vz: 0,
        rotSpeedX: -4.0,
        rotSpeedY: -3.0
      };

      this.activeHalves.push(half1, half2);
      this.particles.emitJuice(pos, innerColor, 30);
    } else {
      // Bomb exploded!
      this.sounds.playBomb();
      this.particles.emitExplosion(pos, 60);
    }

    // Remove original object from scene
    this.scene.remove(obj);
    const idx = this.activeObjects.indexOf(obj);
    if (idx !== -1) {
      this.activeObjects.splice(idx, 1);
    }

    return { isFruit, position: pos, label: obj.userData.label };
  }

  clearAll() {
    for (const obj of this.activeObjects) {
      this.scene.remove(obj);
    }
    this.activeObjects = [];

    for (const half of this.activeHalves) {
      this.scene.remove(half);
    }
    this.activeHalves = [];
  }
}

