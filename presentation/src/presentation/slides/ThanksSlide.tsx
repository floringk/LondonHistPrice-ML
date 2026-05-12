// src/presentation/slides/ThanksSlide.tsx
import type { ThanksSlide as ThanksSlideData } from '../data/slides';

export function ThanksSlide({ data }: { data: ThanksSlideData }) {
  return (
    <div className="slide slide-thanks">
      {data.eyebrow && <div className="thanks-eyebrow">{data.eyebrow}</div>}
      <h1 className="thanks-title">{data.headline}</h1>
      {data.subline && <div className="thanks-subline">{data.subline}</div>}
      {data.context && <div className="thanks-context">{data.context}</div>}
    </div>
  );
}
