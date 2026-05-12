// src/presentation/Presentation.tsx
import { useCallback, useEffect, useState } from 'react';
import { slides, type Slide } from './data/slides';
import { TitleSlide } from './slides/TitleSlide';
import { ContentSlide } from './slides/ContentSlide';
import { TableSlide } from './slides/TableSlide';
import { DualColumnSlide } from './slides/DualColumnSlide';
import { ConclusionSlide } from './slides/ConclusionSlide';
import { ChartSlide } from './slides/ChartSlide';
import { ImageSlide } from './slides/ImageSlide';
import { ThanksSlide } from './slides/ThanksSlide';

function renderSlideContent(slide: Slide) {
  switch (slide.kind) {
    case 'title':
      return <TitleSlide data={slide} />;
    case 'content':
      return <ContentSlide data={slide} />;
    case 'table':
      return <TableSlide data={slide} />;
    case 'dual':
      return <DualColumnSlide data={slide} />;
    case 'conclusion':
      return <ConclusionSlide data={slide} />;
    case 'chart':
      return <ChartSlide data={slide} />;
    case 'image':
      return <ImageSlide data={slide} />;
    case 'thanks':
      return <ThanksSlide data={slide} />;
  }
}

export function Presentation() {
  const [index, setIndex] = useState(0);
  const [fading, setFading] = useState(false);
  const [printMode, setPrintMode] = useState(
    typeof window !== 'undefined' &&
      new URLSearchParams(window.location.search).get('print') === '1'
  );

  const total = slides.length;

  const go = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(total - 1, next));
      if (clamped === index) return;
      setFading(true);
      window.setTimeout(() => {
        setIndex(clamped);
        setFading(false);
      }, 150);
    },
    [index, total]
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (printMode) return;
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
        e.preventDefault();
        go(index + 1);
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault();
        go(index - 1);
      } else if (e.key === 'Home') {
        go(0);
      } else if (e.key === 'End') {
        go(total - 1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [go, index, total, printMode]);

  const exportPDF = async () => {
    setPrintMode(true);
    // Wait two frames for layout to settle, plus a small delay for image decode
    await new Promise<void>(resolve =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    );
    await new Promise(r => window.setTimeout(r, 250));
    window.print();
    setPrintMode(false);
  };

  // Revert print mode if user cancels print dialog via afterprint
  useEffect(() => {
    const onAfterPrint = () => setPrintMode(false);
    window.addEventListener('afterprint', onAfterPrint);
    return () => window.removeEventListener('afterprint', onAfterPrint);
  }, []);

  return (
    <div className={`presentation-root${printMode ? ' print-mode' : ''}`}>
      {printMode ? (
        <div className="print-stack">
          {slides.map((s, i) => (
            <div key={i} className="print-page">
              {renderSlideContent(s)}
            </div>
          ))}
        </div>
      ) : (
        <div className={`slide-stage ${fading ? 'fading' : ''}`}>
          {renderSlideContent(slides[index])}
        </div>
      )}

      <nav className="nav-bar">
        <button
          className="nav-btn"
          onClick={() => go(index - 1)}
          disabled={index === 0}
          aria-label="Previous slide"
        >
          ← Previous
        </button>

        <div className="nav-dots" role="tablist" aria-label="Jump to slide">
          {slides.map((_, i) => (
            <button
              key={i}
              className={`nav-dot ${i === index ? 'active' : ''}`}
              onClick={() => go(i)}
              aria-label={`Go to slide ${i + 1}`}
              aria-current={i === index ? 'true' : 'false'}
            >
              {i + 1}
            </button>
          ))}
        </div>

        <div className="nav-counter">
          Slide <strong>{index + 1}</strong> / {total}
        </div>

        <button
          className="nav-btn nav-btn-export"
          onClick={exportPDF}
          aria-label="Export presentation as PDF"
          title="Open the browser print dialog — choose 'Save as PDF'"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Export PDF
        </button>

        <button
          className="nav-btn"
          onClick={() => go(index + 1)}
          disabled={index === total - 1}
          aria-label="Next slide"
        >
          Next →
        </button>
      </nav>
    </div>
  );
}
