export class BottomControls {
  constructor(container, callbacks = {}) {
    this.callbacks = callbacks;
    this.element = document.createElement('div');
    this.element.id = 'bottom-controls';
    this.element.className = 'glass-panel bottom-bar';
    this.element.innerHTML = `
      <div class="playback-buttons">
        <button id="btn-play-pause" class="ctrl-btn primary-ctrl">▶ Play Training</button>
        <button id="btn-step" class="ctrl-btn">⏭ Step Next</button>
        <button id="btn-reset-train" class="ctrl-btn">↺ Reset Model</button>
      </div>
      <div class="speed-slider-group">
        <label for="slider-speed">Animation Speed:</label>
        <input type="range" id="slider-speed" min="1" max="60" value="20" />
        <span id="speed-label">20x</span>
      </div>
      <div class="shortcuts-hint">
        <span><b>SPACE</b> Freeze</span>
        <span><b>T</b> Train</span>
        <span><b>R</b> Reset</span>
        <span><b>O</b> 3D Orbit</span>
        <span><b>D</b> Dataset</span>
      </div>
    `;
    container.appendChild(this.element);

    this.playBtn = this.element.querySelector('#btn-play-pause');
    this.stepBtn = this.element.querySelector('#btn-step');
    this.resetBtn = this.element.querySelector('#btn-reset-train');
    this.speedSlider = this.element.querySelector('#slider-speed');
    this.speedLabel = this.element.querySelector('#speed-label');

    this.isPlaying = false;

    this.playBtn.addEventListener('click', () => {
      this.isPlaying = !this.isPlaying;
      this.playBtn.textContent = this.isPlaying ? '⏸ Pause Training' : '▶ Play Training';
      this.playBtn.classList.toggle('playing', this.isPlaying);
      if (this.callbacks.onTogglePlay) this.callbacks.onTogglePlay(this.isPlaying);
    });

    this.stepBtn.addEventListener('click', () => {
      if (this.callbacks.onStep) this.callbacks.onStep();
    });

    this.resetBtn.addEventListener('click', () => {
      this.isPlaying = false;
      this.playBtn.textContent = '▶ Play Training';
      this.playBtn.classList.remove('playing');
      if (this.callbacks.onReset) this.callbacks.onReset();
    });

    this.speedSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      this.speedLabel.textContent = `${val}x`;
      if (this.callbacks.onSpeedChange) this.callbacks.onSpeedChange(val);
    });
  }

  setPlaying(isPlaying) {
    this.isPlaying = isPlaying;
    this.playBtn.textContent = isPlaying ? '⏸ Pause Training' : '▶ Play Training';
    this.playBtn.classList.toggle('playing', isPlaying);
  }
}

