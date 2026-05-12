// src/presentation/charts/LossCurveChart.tsx
const data = [
  { epoch: 1,  val_rmse: 514788 },
  { epoch: 2,  val_rmse: 485988 },
  { epoch: 3,  val_rmse: 470722 },
  { epoch: 4,  val_rmse: 466629 },
  { epoch: 5,  val_rmse: 459696 },
  { epoch: 6,  val_rmse: 455007 },
  { epoch: 7,  val_rmse: 471247 },
  { epoch: 8,  val_rmse: 481891 },
  { epoch: 9,  val_rmse: 447266 },
  { epoch: 10, val_rmse: 441493 },
  { epoch: 11, val_rmse: 447548 },
  { epoch: 12, val_rmse: 495560 },
  { epoch: 13, val_rmse: 439696 },
  { epoch: 14, val_rmse: 455208 },
  { epoch: 15, val_rmse: 463789 },
  { epoch: 16, val_rmse: 456313 },
  { epoch: 17, val_rmse: 448985 },
  { epoch: 18, val_rmse: 470632 },
  { epoch: 19, val_rmse: 507828 },
  { epoch: 20, val_rmse: 462431 },
  { epoch: 21, val_rmse: 428185 },
  { epoch: 22, val_rmse: 514633 },
  { epoch: 23, val_rmse: 558572 },
  { epoch: 24, val_rmse: 486197 },
  { epoch: 25, val_rmse: 437970 },
  { epoch: 26, val_rmse: 441506 },
  { epoch: 27, val_rmse: 521232 },
  { epoch: 28, val_rmse: 518817 },
  { epoch: 29, val_rmse: 518999 },
  { epoch: 30, val_rmse: 447508 },
  { epoch: 31, val_rmse: 486591 },
];

const BEST_EPOCH = 21;

const W = 760;
const H = 240;
const PAD_L = 60;
const PAD_R = 24;
const PAD_T = 24;
const PAD_B = 36;
const plotW = W - PAD_L - PAD_R;
const plotH = H - PAD_T - PAD_B;

const yMin = 400000;
const yMax = 580000;
const xMin = 1;
const xMax = 31;

const xScale = (e: number) =>
  PAD_L + ((e - xMin) / (xMax - xMin)) * plotW;
const yScale = (v: number) =>
  PAD_T + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

const points = data.map(d => `${xScale(d.epoch)},${yScale(d.val_rmse)}`).join(' ');
const best = data.find(d => d.epoch === BEST_EPOCH)!;
const fmtK = (v: number) => Math.round(v / 1000) + 'k';

export function LossCurveChart() {
  const yTicks = [400000, 460000, 520000, 580000];
  const xTicks = [1, 5, 10, 15, 20, 25, 31];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="PyTorch MLP validation RMSE per epoch, with best checkpoint marked"
    >
      {/* Y grid + ticks */}
      {yTicks.map(t => (
        <g key={t}>
          <line
            x1={PAD_L} y1={yScale(t)} x2={W - PAD_R} y2={yScale(t)}
            stroke="#334155" strokeWidth={1} strokeDasharray="2 4"
          />
          <text x={PAD_L - 6} y={yScale(t) + 4} textAnchor="end" fontSize={11} fill="#94a3b8">
            {fmtK(t)}
          </text>
        </g>
      ))}
      {/* X ticks */}
      {xTicks.map(t => (
        <text key={t}
          x={xScale(t)}
          y={PAD_T + plotH + 16}
          textAnchor="middle"
          fontSize={11}
          fill="#94a3b8"
        >
          {t}
        </text>
      ))}

      {/* Axis titles */}
      <text x={PAD_L + plotW / 2} y={H - 4} textAnchor="middle" fontSize={12} fill="#94a3b8">
        Epoch
      </text>
      <text
        x={14}
        y={PAD_T + plotH / 2}
        textAnchor="middle"
        fontSize={12}
        fill="#94a3b8"
        transform={`rotate(-90, 14, ${PAD_T + plotH / 2})`}
      >
        Val RMSE (£)
      </text>

      {/* Best checkpoint vertical dashed line */}
      <line
        x1={xScale(BEST_EPOCH)} y1={PAD_T}
        x2={xScale(BEST_EPOCH)} y2={PAD_T + plotH}
        stroke="#10b981" strokeDasharray="4 4" strokeWidth={1.2}
      />

      {/* Polyline */}
      <polyline
        points={points}
        fill="none"
        stroke="#3b82f6"
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* Per-point markers */}
      {data.map(d => (
        <circle
          key={d.epoch}
          cx={xScale(d.epoch)}
          cy={yScale(d.val_rmse)}
          r={d.epoch === BEST_EPOCH ? 6 : 2.5}
          fill={d.epoch === BEST_EPOCH ? '#10b981' : '#60a5fa'}
          stroke={d.epoch === BEST_EPOCH ? '#0f172a' : 'none'}
          strokeWidth={d.epoch === BEST_EPOCH ? 2 : 0}
        />
      ))}

      {/* Best checkpoint label */}
      <g>
        <rect
          x={xScale(BEST_EPOCH) - 92}
          y={yScale(best.val_rmse) - 36}
          width={184}
          height={22}
          fill="#10b981"
          rx={11}
        />
        <text
          x={xScale(BEST_EPOCH)}
          y={yScale(best.val_rmse) - 21}
          textAnchor="middle"
          fontSize={11.5}
          fontWeight={700}
          fill="#0f172a"
        >
          Best checkpoint · 428,185 £
        </text>
      </g>
    </svg>
  );
}
