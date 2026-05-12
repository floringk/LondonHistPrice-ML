// src/presentation/slides/TitleSlide.tsx
import type { TitleSlide as TitleSlideData } from '../data/slides';

export function TitleSlide({ data }: { data: TitleSlideData }) {
  return (
    <div className="slide slide-title">
      <div className="title-eyebrow">London Housing AVM</div>
      <h1 className="title-main">{data.mainTitle}</h1>
      <h2 className="title-sub">{data.subtitle}</h2>
      <div className="title-tagline">{data.tagline}</div>
      {data.narrativeContext && (
        <div className="narrative-context">{data.narrativeContext}</div>
      )}
      <div className="title-context">{data.context}</div>
    </div>
  );
}
