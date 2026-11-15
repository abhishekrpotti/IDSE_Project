import * as THREE from 'three';

/**
 * Procedural Fruit Meshes for ML Ninja
 * Class +1 objects: Watermelon, Orange, Apple, Kiwi, Pineapple
 */

const FRUIT_TYPES = ['watermelon', 'orange', 'apple', 'kiwi', 'pineapple'];

export function createFruitMesh(fruitType = 'watermelon') {
  const group = new THREE.Group();
  group.userData = { type: 'fruit', fruitType, isSliced: false };

  let mainGeometry;
  let mainMaterial;

  switch (fruitType) {
    case 'watermelon': {
      mainGeometry = new THREE.SphereGeometry(0.55, 24, 20);
      mainGeometry.scale(1.2, 0.95, 0.95);
      mainMaterial = new THREE.MeshStandardMaterial({
        color: 0x2e7d32,
        roughness: 0.35,
        metalness: 0.1
      });
      break;
    }
    case 'orange': {
      mainGeometry = new THREE.SphereGeometry(0.48, 22, 18);
      mainMaterial = new THREE.MeshStandardMaterial({
        color: 0xff9800,
        roughness: 0.45,
        metalness: 0.05
      });
      break;
    }
    case 'apple': {
      mainGeometry = new THREE.SphereGeometry(0.48, 22, 18);
      mainGeometry.scale(1.0, 0.9, 1.0);
      mainMaterial = new THREE.MeshStandardMaterial({
        color: 0xe53935,
        roughness: 0.25,
        metalness: 0.15
      });
      break;
    }
    case 'kiwi': {
      mainGeometry = new THREE.SphereGeometry(0.42, 20, 16);
      mainGeometry.scale(0.85, 1.15, 0.85);
      mainMaterial = new THREE.MeshStandardMaterial({
        color: 0x795548,
        roughness: 0.8,
        metalness: 0.0
      });
      break;
    }
    case 'pineapple':
    default: {
      mainGeometry = new THREE.CylinderGeometry(0.35, 0.42, 0.9, 18);
      mainMaterial = new THREE.MeshStandardMaterial({
        color: 0xfbc02d,
        roughness: 0.5,
        metalness: 0.1
      });
      break;
    }
  }

  const fruitMesh = new THREE.Mesh(mainGeometry, mainMaterial);
  fruitMesh.castShadow = true;
  group.add(fruitMesh);

  // Small stem / leaf for decoration
  const stemGeom = new THREE.CylinderGeometry(0.04, 0.03, 0.18, 6);
  const stemMat = new THREE.MeshStandardMaterial({ color: 0x33691e });
  const stem = new THREE.Mesh(stemGeom, stemMat);
  stem.position.y = 0.52;
  group.add(stem);

  // Bounding radius for fast slicing collision
  group.userData.radius = 0.55;

  return group;
}

/**
 * Creates two separated hemisphere halves with juicy pulp interior
 */
export function createSlicedHalves(fruitType = 'watermelon') {
  const group = new THREE.Group();

  let outerColor = 0x2e7d32;
  let innerColor = 0xe53935; // watermelon pink/red inside

  if (fruitType === 'orange') {
    outerColor = 0xff9800;
    innerColor = 0xffb74d;
  } else if (fruitType === 'apple') {
    outerColor = 0xe53935;
    innerColor = 0xfff9c4;
  } else if (fruitType === 'kiwi') {
    outerColor = 0x795548;
    innerColor = 0x8bc34a;
  } else if (fruitType === 'pineapple') {
    outerColor = 0xfbc02d;
    innerColor = 0xffeb3b;
  }

  const outerMat = new THREE.MeshStandardMaterial({ color: outerColor, roughness: 0.4 });
  const innerMat = new THREE.MeshStandardMaterial({ color: innerColor, roughness: 0.3 });

  // Half 1: upper hemisphere
  const halfGeom1 = new THREE.SphereGeometry(0.5, 20, 16, 0, Math.PI * 2, 0, Math.PI / 2);
  const capGeom = new THREE.CircleGeometry(0.5, 20);
  capGeom.rotateX(Math.PI / 2);

  const half1 = new THREE.Group();
  half1.add(new THREE.Mesh(halfGeom1, outerMat));
  half1.add(new THREE.Mesh(capGeom, innerMat));

  // Half 2: lower hemisphere
  const halfGeom2 = new THREE.SphereGeometry(0.5, 20, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
  const half2 = new THREE.Group();
  half2.add(new THREE.Mesh(halfGeom2, outerMat));
  const cap2 = new THREE.Mesh(capGeom, innerMat);
  cap2.rotateX(Math.PI);
  half2.add(cap2);

  group.add(half1);
  group.add(half2);

  return { group, half1, half2, innerColor };
}

