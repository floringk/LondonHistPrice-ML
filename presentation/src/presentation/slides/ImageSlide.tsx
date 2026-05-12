// src/presentation/slides/ImageSlide.tsx
import type { ImageSlide as ImageSlideData } from '../data/slides';

export function ImageSlide({ data }: { data: ImageSlideData }) {
  const single = data.images.length === 1;
  return (
    <div className="slide slide-image">
      <header className="slide-header">
        <h1>{data.title}</h1>
        {data.subtitle && <p className="slide-subtitle">{data.subtitle}</p>}
      </header>

      {data.chips && data.chips.length > 0 && (
        <div className="chip-stack chip-stack-top">
          {data.chips.map((c, i) => (
            <div key={i} className="explain-chip">
              <span className="chip-icon" aria-hidden="true">ℹ</span>
              <span>{c}</span>
            </div>
          ))}
        </div>
      )}

      <div className={single ? 'image-grid image-grid-single' : 'image-grid image-grid-pair'}>
        {data.images.map((img, i) => (
          <figure key={i} className="image-card">
            <img src={img.src} alt={img.alt} loading="lazy" />
            {img.caption && <figcaption>{img.caption}</figcaption>}
            {img.chip && (
              <div className="explain-chip image-chip">
                <span className="chip-icon" aria-hidden="true">ℹ</span>
                <span>{img.chip}</span>
              </div>
            )}
          </figure>
        ))}
      </div>

      {data.badges && data.badges.length > 0 && (
        <div className="badge-row">
          {data.badges.map((b, i) => (
            <div key={i} className="badge">
              <span className="badge-label">{b.label}</span>
              <span className="badge-value">{b.value}</span>
            </div>
          ))}
        </div>
      )}

      {data.footnote && <div className="footnote">{data.footnote}</div>}

      {data.bridge && <div className="slide-bridge">→ {data.bridge}</div>}
    </div>
  );
}
