import * as THREE from 'three';

export class PhysicsSystem {
  constructor() {
    this.gravity = -8.5; // World units/s^2
  }

  updateObject(obj, dt) {
    if (!obj.userData.physics) return;
    const phys = obj.userData.physics;

    if (phys.isFrozen) return;

    // Position integration
    obj.position.x += phys.vx * dt;
    obj.position.y += phys.vy * dt;
    obj.position.z += phys.vz * dt;

    // Gravity
    phys.vy += this.gravity * dt;

    // Rotation
    obj.rotation.x += phys.rotSpeedX * dt;
    obj.rotation.y += phys.rotSpeedY * dt;
    obj.rotation.z += phys.rotSpeedZ * dt;
  }

  updateHalf(halfMesh, dt) {
    if (!halfMesh.userData.physics) return;
    const phys = halfMesh.userData.physics;
    halfMesh.position.x += phys.vx * dt;
    halfMesh.position.y += phys.vy * dt;
    halfMesh.position.z += phys.vz * dt;

    phys.vy += this.gravity * dt;

    halfMesh.rotation.x += phys.rotSpeedX * dt;
    halfMesh.rotation.y += phys.rotSpeedY * dt;
  }
}

