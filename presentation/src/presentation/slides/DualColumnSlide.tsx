// src/presentation/slides/DualColumnSlide.tsx
import type { DualSlide as DualSlideData } from '../data/slides';

export function DualColumnSlide({ data }: { data: DualSlideData }) {
  return (
    <div className="slide slide-dual">
      <header className="slide-header">
        <h1>{data.title}</h1>
        {data.subtitle && <p className="slide-subtitle">{data.subtitle}</p>}
      </header>
      <div className="dual-grid">
        <section className={`dual-col tone-${data.left.tone ?? 'positive'}`}>
          <h2>{data.left.heading}</h2>
          <ul>
            {data.left.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
          {data.left.chip && (
            <div className="explain-chip dual-col-chip">
              <span className="chip-icon" aria-hidden="true">ℹ</span>
              <span>{data.left.chip}</span>
            </div>
          )}
        </section>
        <section className={`dual-col tone-${data.right.tone ?? 'negative'}`}>
          <h2>{data.right.heading}</h2>
          <ul>
            {data.right.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ul>
          {data.right.chip && (
            <div className="explain-chip dual-col-chip">
              <span className="chip-icon" aria-hidden="true">ℹ</span>
              <span>{data.right.chip}</span>
            </div>
          )}
        </section>
      </div>

      {data.bridge && <div className="slide-bridge">→ {data.bridge}</div>}
    </div>
  );
}
