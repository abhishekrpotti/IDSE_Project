export class TooltipController {
  constructor(container) {
    this.el = document.createElement('div');
    this.el.id = 'ml-tooltip';
    this.el.className = 'glass-tooltip';
    this.el.style.display = 'none';
    container.appendChild(this.el);
  }

  show(clientX, clientY, data) {
    this.el.innerHTML = `
      <div class="tooltip-header">${data.type === 'fruit' ? '🍎 Fruit (+1)' : '💣 Bomb (-1)'}</div>
      <div class="tooltip-row"><span>Feature x₁:</span> <b>${data.x.toFixed(2)}</b></div>
      <div class="tooltip-row"><span>Feature x₂:</span> <b>${data.y.toFixed(2)}</b></div>
      ${data.predictedLabel !== undefined ? `
        <div class="tooltip-row"><span>Predicted:</span> <b>${data.predictedLabel >= 0 ? 'Fruit' : 'Bomb'}</b></div>
        <div class="tooltip-row"><span>Confidence:</span> <b>${(data.confidence * 100).toFixed(1)}%</b></div>
      ` : ''}
    `;
    this.el.style.left = `${clientX + 14}px`;
    this.el.style.top = `${clientY + 14}px`;
    this.el.style.display = 'block';
  }

  hide() {
    this.el.style.display = 'none';
  }
}

