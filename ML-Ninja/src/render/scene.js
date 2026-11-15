import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export function createSceneSetup(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f111a); // Dark sleek background

  // Camera looking at origin with slight perspective tilt
  const camera = new THREE.PerspectiveCamera(
    48,
    window.innerWidth / window.innerHeight,
    0.1,
    100
  );
  camera.position.set(0, 0.4, 9.5);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance'
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(4, 8, 6);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 1024;
  dirLight.shadow.mapSize.height = 1024;
  scene.add(dirLight);

  // Secondary soft colored rim lights for futuristic arcade vibe
  const rimCyan = new THREE.PointLight(0x00e5ff, 0.8, 15);
  rimCyan.position.set(-6, -2, 3);
  scene.add(rimCyan);

  const rimMagenta = new THREE.PointLight(0xff007f, 0.6, 15);
  rimMagenta.position.set(6, -2, 3);
  scene.add(rimMagenta);

  // Orbit controls (disabled by default for slicing, toggled via 'O')
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enabled = false; // Disabled by default during arcade slicing

  const originalCamPos = camera.position.clone();
  const originalCamRot = camera.rotation.clone();

  function toggleOrbit() {
    controls.enabled = !controls.enabled;
    if (!controls.enabled) {
      // Reset camera to front facing slice plane
      camera.position.copy(originalCamPos);
      camera.rotation.copy(originalCamRot);
      controls.target.set(0, 0, 0);
    }
    return controls.enabled;
  }

  function handleResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  window.addEventListener('resize', handleResize);

  return {
    scene,
    camera,
    renderer,
    controls,
    toggleOrbit,
    handleResize
  };
}

