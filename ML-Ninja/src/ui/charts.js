import Chart from 'chart.js/auto';

export class ChartPanels {
  constructor(container) {
    this.element = document.createElement('div');
    this.element.id = 'charts-panel';
    this.element.className = 'glass-panel';
    this.element.innerHTML = `
      <div class="chart-box">
        <div class="chart-title">TRAINING METRICS</div>
        <div style="position: relative; height: 110px; width: 100%;">
          <canvas id="lossChartCanvas"></canvas>
        </div>
      </div>
      <div class="chart-box">
        <div class="chart-title">CONFUSION MATRIX</div>
        <div class="confusion-grid">
          <div class="cm-cell cm-header"></div>
          <div class="cm-cell cm-header">Pred Fruit</div>
          <div class="cm-cell cm-header">Pred Bomb</div>
          <div class="cm-cell cm-header">True Fruit</div>
          <div class="cm-cell cm-tp" id="cm-tp">0</div>
          <div class="cm-cell cm-fn" id="cm-fn">0</div>
          <div class="cm-cell cm-header">True Bomb</div>
          <div class="cm-cell cm-fp" id="cm-fp">0</div>
          <div class="cm-cell cm-tn" id="cm-tn">0</div>
        </div>
      </div>
    `;
    container.appendChild(this.element);

    this.tpEl = this.element.querySelector('#cm-tp');
    this.fnEl = this.element.querySelector('#cm-fn');
    this.fpEl = this.element.querySelector('#cm-fp');
    this.tnEl = this.element.querySelector('#cm-tn');

    const ctx = this.element.querySelector('#lossChartCanvas').getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Accuracy',
            data: [],
            borderColor: '#00e676',
            backgroundColor: 'rgba(0, 230, 118, 0.1)',
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 0
          },
          {
            label: 'Loss',
            data: [],
            borderColor: '#ff5252',
            backgroundColor: 'rgba(255, 82, 82, 0.1)',
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { display: false },
          y: {
            display: true,
            min: 0,
            max: 1.2,
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            ticks: { color: '#90a4ae', font: { size: 9 } }
          }
        },
        plugins: {
          legend: {
            display: true,
            labels: { color: '#cfd8dc', boxWidth: 10, font: { size: 9 } }
          }
        }
      }
    });
  }

  reset() {
    this.chart.data.labels = [];
    this.chart.data.datasets[0].data = [];
    this.chart.data.datasets[1].data = [];
    this.chart.update();
    this.tpEl.textContent = '0';
    this.fnEl.textContent = '0';
    this.fpEl.textContent = '0';
    this.tnEl.textContent = '0';
  }

  updateMetrics(accuracy, loss, step) {
    this.chart.data.labels.push(step);
    this.chart.data.datasets[0].data.push(accuracy ?? 0);
    this.chart.data.datasets[1].data.push(loss !== undefined ? Math.min(1.2, loss) : null);

    // Limit to latest 50 points
    if (this.chart.data.labels.length > 50) {
      this.chart.data.labels.shift();
      this.chart.data.datasets[0].data.shift();
      this.chart.data.datasets[1].data.shift();
    }
    this.chart.update();
  }

  updateConfusionMatrix(matrix) {
    if (!matrix) return;
    // matrix: [[tn, fp], [fn, tp]]
    const [[tn, fp], [fn, tp]] = matrix;
    this.tpEl.textContent = tp;
    this.fnEl.textContent = fn;
    this.fpEl.textContent = fp;
    this.tnEl.textContent = tn;
  }
}

