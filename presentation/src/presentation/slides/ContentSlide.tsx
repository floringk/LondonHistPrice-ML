// src/presentation/slides/ContentSlide.tsx
import type { ContentSlide as ContentSlideData } from '../data/slides';
import { ChartById } from '../charts';

export function ContentSlide({ data }: { data: ContentSlideData }) {
  return (
    <div className="slide slide-content">
      <header className="slide-header">
        <h1>{data.title}</h1>
        {data.subtitle && <p className="slide-subtitle">{data.subtitle}</p>}
      </header>
      <ul className="bullet-list">
        {data.bullets.map((b, i) => (
          <li key={i} className={b.highlight ? 'bullet bullet-highlight' : 'bullet'}>
            <div className="bullet-body">
              <span className="bullet-marker">●</span>
              <span className="bullet-text">{b.text}</span>
            </div>
            {b.chip && (
              <div className="explain-chip bullet-chip">
                <span className="chip-icon" aria-hidden="true">ℹ</span>
                <span>{b.chip}</span>
              </div>
            )}
          </li>
        ))}
      </ul>
      {data.callout && <div className="callout">{data.callout}</div>}
      {data.footnote && <div className="footnote">{data.footnote}</div>}

      {data.chartId && (
        <div className="chart-container chart-container-inline">
          <ChartById id={data.chartId} />
        </div>
      )}

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
