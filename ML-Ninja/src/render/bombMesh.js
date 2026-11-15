import * as THREE from 'three';

export function createBombMesh() {
  const group = new THREE.Group();
  group.userData = { type: 'bomb', isSliced: false, radius: 0.5 };

  // Main iron sphere
  const sphereGeom = new THREE.SphereGeometry(0.48, 24, 20);
  const sphereMat = new THREE.MeshStandardMaterial({
    color: 0x1a1a1a,
    roughness: 0.35,
    metalness: 0.8
  });
  const body = new THREE.Mesh(sphereGeom, sphereMat);
  body.castShadow = true;
  group.add(body);

  // Metal cap
  const capGeom = new THREE.CylinderGeometry(0.12, 0.16, 0.12, 12);
  const capMat = new THREE.MeshStandardMaterial({ color: 0x757575, metalness: 0.9, roughness: 0.2 });
  const cap = new THREE.Mesh(capGeom, capMat);
  cap.position.y = 0.46;
  group.add(cap);

  // Fuse
  const curve = new THREE.CubicBezierCurve3(
    new THREE.Vector3(0, 0.5, 0),
    new THREE.Vector3(0.08, 0.65, 0.05),
    new THREE.Vector3(-0.06, 0.75, -0.05),
    new THREE.Vector3(0.05, 0.88, 0)
  );
  const fuseGeom = new THREE.TubeGeometry(curve, 16, 0.035, 8, false);
  const fuseMat = new THREE.MeshStandardMaterial({ color: 0x8d6e63, roughness: 0.9 });
  const fuse = new THREE.Mesh(fuseGeom, fuseMat);
  group.add(fuse);

  // Pulsing red hazard LED
  const ledGeom = new THREE.SphereGeometry(0.08, 12, 12);
  const ledMat = new THREE.MeshBasicMaterial({ color: 0xff1744 });
  const led = new THREE.Mesh(ledGeom, ledMat);
  led.position.set(0.05, 0.88, 0);
  group.add(led);

  // Spark light at fuse tip
  const sparkLight = new THREE.PointLight(0xff3d00, 1.8, 2.5);
  sparkLight.position.set(0.05, 0.88, 0);
  group.add(sparkLight);

  group.userData.ledMat = ledMat;
  group.userData.sparkLight = sparkLight;

  return group;
}

