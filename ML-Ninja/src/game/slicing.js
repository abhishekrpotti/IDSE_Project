import * as THREE from 'three';
import { lineFromSegment } from '../utils/math.js';

export class SlicingController {
  constructor(camera, planeZ = 0) {
    this.camera = camera;
    this.planeZ = planeZ;
    this.plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -planeZ);
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.lastWorldPos = null;
    this.currentWorldPos = new THREE.Vector3();
    this.isPointerDown = false;

    // Track full slash stroke for hand-drawn boundary
    this.slashStrokePoints = [];
    this.lastSlashBoundary = null;
  }

  getPointerWorldPos(clientX, clientY) {
    this.mouse.x = (clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(clientY / window.innerHeight) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const target = new THREE.Vector3();
    this.raycaster.ray.intersectPlane(this.plane, target);
    return target;
  }

  startSlash(clientX, clientY) {
    this.isPointerDown = true;
    const worldPos = this.getPointerWorldPos(clientX, clientY);
    this.lastWorldPos = worldPos.clone();
    this.currentWorldPos.copy(worldPos);
    this.slashStrokePoints = [worldPos.clone()];
  }

  moveSlash(clientX, clientY) {
    const worldPos = this.getPointerWorldPos(clientX, clientY);
    this.currentWorldPos.copy(worldPos);
    if (this.isPointerDown) {
      this.slashStrokePoints.push(worldPos.clone());
    }
  }

  endSlash() {
    this.isPointerDown = false;
    this.lastWorldPos = null;

    // If stroke was a decisive slash line (length > 2 units), calculate player boundary line
    if (this.slashStrokePoints.length >= 2) {
      const pStart = this.slashStrokePoints[0];
      const pEnd = this.slashStrokePoints[this.slashStrokePoints.length - 1];
      const dist = Math.hypot(pEnd.x - pStart.x, pEnd.y - pStart.y);
      if (dist >= 1.5) {
        this.lastSlashBoundary = lineFromSegment([pStart.x, pStart.y], [pEnd.x, pEnd.y]);
        this.lastSlashBoundary.pStart = pStart;
        this.lastSlashBoundary.pEnd = pEnd;
      }
    }
    this.slashStrokePoints = [];
  }

  /**
   * Checks if segment (p1 -> p2) intersects sphere centered at c with radius r
   */
  checkSegmentSphereIntersection(p1, p2, c, r) {
    const v = new THREE.Vector3().subVectors(p2, p1);
    const len = v.length();
    if (len < 1e-5) {
      return p1.distanceTo(c) <= r;
    }
    v.normalize();
    const w = new THREE.Vector3().subVectors(c, p1);
    const proj = w.dot(v);
    const clamped = Math.max(0, Math.min(len, proj));
    const closest = new THREE.Vector3().copy(p1).addScaledVector(v, clamped);
    return closest.distanceTo(c) <= r;
  }

  /**
   * Slices active targets
   */
  sliceCheck(targets) {
    if (!this.lastWorldPos) {
      this.lastWorldPos = this.currentWorldPos.clone();
      return [];
    }

    const p1 = this.lastWorldPos;
    const p2 = this.currentWorldPos;
    const sliced = [];

    const strokeSpeed = p1.distanceTo(p2);
    // Only slice if blade is moving fast enough
    if (strokeSpeed > 0.05) {
      for (const target of targets) {
        if (target.userData.isSliced) continue;
        const radius = target.userData.radius ?? 0.5;
        const isHit = this.checkSegmentSphereIntersection(p1, p2, target.position, radius);
        if (isHit) {
          sliced.push({ target, cutNormal: new THREE.Vector3(p2.y - p1.y, -(p2.x - p1.x), 0).normalize() });
        }
      }
    }

    this.lastWorldPos.copy(this.currentWorldPos);
    return sliced;
  }
}

