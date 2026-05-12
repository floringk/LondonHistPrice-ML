// src/presentation/slides/ChartSlide.tsx
import type { ChartSlide as ChartSlideData } from '../data/slides';
import { ChartById } from '../charts';

export function ChartSlide({ data }: { data: ChartSlideData }) {
  return (
    <div className="slide slide-chart">
      <header className="slide-header">
        <h1>{data.title}</h1>
        {data.subtitle && <p className="slide-subtitle">{data.subtitle}</p>}
      </header>
      <div className="chart-container chart-container-large">
        <ChartById id={data.chartId} />
      </div>
      {data.caption && <div className="chart-caption">{data.caption}</div>}

      {data.chips && data.chips.length > 0 && (
        <div className="chip-stack">
          {data.chips.map((c, i) => (
            <div key={i} className="explain-chip">
              <span className="chip-icon" aria-hidden="true">ℹ</span>
              <span>{c}</span>
            </div>
          ))}
        </div>
      )}

      {data.bridge && <div className="slide-bridge">→ {data.bridge}</div>}
    </div>
  );
}
