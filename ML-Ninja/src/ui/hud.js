export class HUD {
  constructor(container) {
    this.element = document.createElement('div');
    this.element.id = 'hud-top';
    this.element.innerHTML = `
      <div class="hud-left">
        <div class="hud-brand"><span class="brand-ml">ML</span> <span class="brand-ninja">NINJA</span></div>
        <div class="mode-badge" id="hud-mode-badge">FLOW MODE</div>
      </div>
      <div class="hud-center">
        <div class="stat-box">
          <span class="stat-label">SCORE</span>
          <span class="stat-value" id="hud-score">0</span>
        </div>
        <div class="stat-box">
          <span class="stat-label">LIVES</span>
          <div class="lives-container" id="hud-lives">
            <span class="heart active">❤️</span>
            <span class="heart active">❤️</span>
            <span class="heart active">❤️</span>
          </div>
        </div>
        <div class="stat-box">
          <span class="stat-label">COMBO</span>
          <span class="stat-value combo-val" id="hud-combo">x0</span>
        </div>
      </div>
      <div class="hud-right">
        <div class="stat-box">
          <span class="stat-label">MODEL ACCURACY</span>
          <span class="stat-value acc-val" id="hud-acc">--</span>
        </div>
        <div class="stat-box" id="hud-player-box" style="display: none;">
          <span class="stat-label">YOUR SLASH ACC</span>
          <span class="stat-value player-acc-val" id="hud-player-acc">--</span>
        </div>
        <button id="btn-sound-toggle" class="hud-btn" title="Toggle Sound">🔊</button>
      </div>
    `;
    container.appendChild(this.element);

    this.scoreEl = this.element.querySelector('#hud-score');
    this.livesEl = this.element.querySelector('#hud-lives');
    this.comboEl = this.element.querySelector('#hud-combo');
    this.accEl = this.element.querySelector('#hud-acc');
    this.modeEl = this.element.querySelector('#hud-mode-badge');
    this.playerBoxEl = this.element.querySelector('#hud-player-box');
    this.playerAccEl = this.element.querySelector('#hud-player-acc');
    this.soundBtn = this.element.querySelector('#btn-sound-toggle');
  }

  update(state) {
    this.scoreEl.textContent = state.score;
    this.comboEl.textContent = `x${state.combo}`;
    if (state.combo > 1) {
      this.comboEl.classList.add('combo-pulse');
    } else {
      this.comboEl.classList.remove('combo-pulse');
    }

    // Update hearts
    const hearts = this.livesEl.querySelectorAll('.heart');
    hearts.forEach((h, i) => {
      if (i < state.lives) {
        h.classList.add('active');
        h.classList.remove('dead');
      } else {
        h.classList.remove('active');
        h.classList.add('dead');
      }
    });

    if (state.accuracy !== null && state.accuracy !== undefined) {
      this.accEl.textContent = `${(state.accuracy * 100).toFixed(1)}%`;
      this.accEl.style.color = state.accuracy >= 0.85 ? '#00e676' : (state.accuracy >= 0.65 ? '#ffeb3b' : '#ff5252');
    }

    if (state.playerAccuracy !== null && state.playerAccuracy !== undefined) {
      this.playerBoxEl.style.display = 'flex';
      this.playerAccEl.textContent = `${(state.playerAccuracy * 100).toFixed(1)}%`;
    }

    if (state.mode) {
      this.modeEl.textContent = `${state.mode} MODE`;
      this.modeEl.className = `mode-badge mode-${state.mode.toLowerCase()}`;
    }
  }
}

