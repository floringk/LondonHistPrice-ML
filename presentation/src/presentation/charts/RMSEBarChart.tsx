// src/presentation/charts/RMSEBarChart.tsx
const data = [
  { name: 'HistGBR',       rmse: 393489, role: 'winner' as const },
  { name: 'RandomForest',  rmse: 396947, role: 'normal' as const },
  { name: 'Ridge',         rmse: 562968, role: 'normal' as const },
  { name: 'ElasticNet',    rmse: 563113, role: 'normal' as const },
  { name: 'MLP_medium',    rmse: 645104, role: 'normal' as const },
  { name: 'MLP_large',     rmse: 721999, role: 'normal' as const },
  { name: 'TorchMLP',      rmse: 729943, role: 'normal' as const },
  { name: 'MLP_small',     rmse: 838729, role: 'normal' as const },
  { name: 'NaiveMedian',   rmse: 927646, role: 'baseline' as const },
];

const W = 900;
const H = 420;
const PAD_L = 130;
const PAD_R = 100;
const PAD_T = 16;
const PAD_B = 36;
const plotW = W - PAD_L - PAD_R;
const plotH = H - PAD_T - PAD_B;

const maxRmse = Math.max(...data.map(d => d.rmse));
const xScale = (v: number) => (v / maxRmse) * plotW;
const barH = plotH / data.length - 6;

const color = (role: 'winner' | 'normal' | 'baseline') =>
  role === 'winner' ? '#3b82f6'
  : role === 'baseline' ? '#ef4444'
  : '#475569';

const fmt = (n: number) => n.toLocaleString('en-GB') + ' £';

export function RMSEBarChart() {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Horizontal bar chart of test RMSE across nine models"
    >
      {/* axis baseline */}
      <line
        x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH}
        stroke="#475569" strokeWidth={1}
      />

      {/* NaiveMedian reference dashed vertical line */}
      <line
        x1={PAD_L + xScale(927646)}
        y1={PAD_T}
        x2={PAD_L + xScale(927646)}
        y2={PAD_T + plotH}
        stroke="#ef4444"
        strokeDasharray="4 4"
        strokeWidth={1}
        opacity={0.55}
      />
      <text
        x={PAD_L + xScale(927646) - 4}
        y={PAD_T + 12}
        textAnchor="end"
        fontSize={11}
        fill="#ef4444"
      >
        Naive ref
      </text>

      {data.map((d, i) => {
        const y = PAD_T + i * (plotH / data.length) + 3;
        const w = xScale(d.rmse);
        return (
          <g key={d.name}>
            <text
              x={PAD_L - 10}
              y={y + barH / 2 + 4}
              textAnchor="end"
              fontSize={13}
              fill={d.role === 'winner' ? '#60a5fa' : '#e2e8f0'}
              fontWeight={d.role === 'winner' ? 700 : 500}
            >
              {d.name}{d.role === 'winner' ? ' ✓' : ''}
            </text>
            <rect
              x={PAD_L}
              y={y}
              width={w}
              height={barH}
              fill={color(d.role)}
              opacity={d.role === 'winner' ? 1 : d.role === 'baseline' ? 0.85 : 0.7}
              rx={3}
            />
            <text
              x={PAD_L + w + 8}
              y={y + barH / 2 + 4}
              fontSize={12}
              fill={d.role === 'winner' ? '#60a5fa' : '#cbd5e1'}
              fontWeight={d.role === 'winner' ? 700 : 500}
            >
              {fmt(d.rmse)}
            </text>
          </g>
        );
      })}

      {/* x-axis title */}
      <text
        x={PAD_L + plotW / 2}
        y={H - 6}
        textAnchor="middle"
        fontSize={12}
        fill="#94a3b8"
      >
        Test RMSE (£) — sorted ascending
      </text>
    </svg>
  );
}
