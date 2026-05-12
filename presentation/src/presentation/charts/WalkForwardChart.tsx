// src/presentation/charts/WalkForwardChart.tsx
const data = [
  { fold: '2000→2006', rf: 224291, mlp: 293533, winner: 'rf' as const },
  { fold: '2006→2012', rf: 341328, mlp: 312846, winner: 'mlp' as const },
  { fold: '2012→2018', rf: 462977, mlp: 411279, winner: 'mlp' as const },
  { fold: '2018→2024', rf: 361405, mlp: 408975, winner: 'rf' as const },
];

const W = 760;
const H = 260;
const PAD_L = 60;
const PAD_R = 16;
const PAD_T = 36;
const PAD_B = 60;
const plotW = W - PAD_L - PAD_R;
const plotH = H - PAD_T - PAD_B;

const allVals = data.flatMap(d => [d.rf, d.mlp]);
const yMax = Math.ceil(Math.max(...allVals) / 100000) * 100000;
const yScale = (v: number) => PAD_T + plotH - (v / yMax) * plotH;

const groupW = plotW / data.length;
const barW = (groupW - 16) / 2;

const RF_COLOR = '#3b82f6';
const MLP_COLOR = '#f59e0b';

const fmt = (n: number) => Math.round(n / 1000) + 'k';

export function WalkForwardChart() {
  const yTicks = [0, yMax * 0.25, yMax * 0.5, yMax * 0.75, yMax];

  return (
    <div className="chart-wrap">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Grouped bar chart of RMSE per walk-forward fold for RandomForest vs MLP"
      >
        {/* Legend */}
        <g transform={`translate(${PAD_L}, 12)`}>
          <rect x={0} y={0} width={14} height={14} fill={RF_COLOR} rx={2} />
          <text x={20} y={11} fontSize={12} fill="#e2e8f0">RandomForest</text>
          <rect x={130} y={0} width={14} height={14} fill={MLP_COLOR} rx={2} />
          <text x={150} y={11} fontSize={12} fill="#e2e8f0">MLP (sklearn)</text>
        </g>

        {/* Y axis grid + ticks */}
        {yTicks.map(t => {
          const y = yScale(t);
          return (
            <g key={t}>
              <line
                x1={PAD_L} y1={y} x2={W - PAD_R} y2={y}
                stroke="#334155" strokeWidth={1} strokeDasharray={t === 0 ? '' : '2 4'}
              />
              <text x={PAD_L - 6} y={y + 4} textAnchor="end" fontSize={11} fill="#94a3b8">
                {fmt(t)}
              </text>
            </g>
          );
        })}
        <text
          x={14}
          y={PAD_T + plotH / 2}
          textAnchor="middle"
          fontSize={12}
          fill="#94a3b8"
          transform={`rotate(-90, 14, ${PAD_T + plotH / 2})`}
        >
          RMSE (£)
        </text>

        {/* Bars */}
        {data.map((d, i) => {
          const groupX = PAD_L + i * groupW + 8;
          const rfH = plotH - (yScale(d.rf) - PAD_T);
          const mlpH = plotH - (yScale(d.mlp) - PAD_T);
          return (
            <g key={d.fold}>
              <rect
                x={groupX}
                y={yScale(d.rf)}
                width={barW}
                height={rfH}
                fill={RF_COLOR}
                rx={2}
              />
              <rect
                x={groupX + barW + 4}
                y={yScale(d.mlp)}
                width={barW}
                height={mlpH}
                fill={MLP_COLOR}
                rx={2}
              />
              <text
                x={groupX + groupW / 2 - 8}
                y={PAD_T + plotH + 18}
                textAnchor="middle"
                fontSize={12}
                fill="#e2e8f0"
                fontWeight={500}
              >
                {d.fold}
              </text>
              {d.winner === 'mlp' && (
                <g>
                  <rect
                    x={groupX + barW - 6}
                    y={yScale(d.mlp) - 22}
                    width={70}
                    height={18}
                    fill="#f59e0b"
                    rx={9}
                  />
                  <text
                    x={groupX + barW + 28}
                    y={yScale(d.mlp) - 9}
                    textAnchor="middle"
                    fontSize={10.5}
                    fill="#0f172a"
                    fontWeight={700}
                  >
                    MLP wins
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
      <div className="cv-summary">
        <div className="cv-item"><span className="cv-label">RF CV</span><span className="cv-value">24.4%</span></div>
        <div className="cv-item cv-winner"><span className="cv-label">MLP CV</span><span className="cv-value">15.1% ✓</span></div>
      </div>
    </div>
  );
}
