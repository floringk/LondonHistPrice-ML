// src/presentation/slides/ConclusionSlide.tsx
import type { ConclusionSlide as ConclusionSlideData } from '../data/slides';

export function ConclusionSlide({ data }: { data: ConclusionSlideData }) {
  return (
    <div className="slide slide-conclusion">
      <header className="slide-header">
        <h1>{data.title}</h1>
      </header>

      {data.summaryChip && (
        <div className="summary-chip">
          <span className="chip-icon" aria-hidden="true">ℹ</span>
          <span>{data.summaryChip}</span>
        </div>
      )}

      <ol className="conclusion-list">
        {data.items.map((it, i) => (
          <li key={i} className="conclusion-item">
            <span className="conclusion-tag">{it.tag}</span>
            <span className="conclusion-text">{it.text}</span>
          </li>
        ))}
      </ol>
      {data.finalNote && <div className="final-note">{data.finalNote}</div>}
    </div>
  );
}
