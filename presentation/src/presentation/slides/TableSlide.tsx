// src/presentation/slides/TableSlide.tsx
import { Fragment } from 'react';
import type { TableSlide as TableSlideData } from '../data/slides';
import { ChartById } from '../charts';

export function TableSlide({ data }: { data: TableSlideData }) {
  return (
    <div className="slide slide-table">
      <header className="slide-header">
        <h1>{data.title}</h1>
        {data.subtitle && <p className="slide-subtitle">{data.subtitle}</p>}
      </header>

      <table className="results-table">
        <thead>
          <tr>
            {data.headers.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((r, i) => (
            <Fragment key={i}>
              <tr className={r.winner ? 'row-winner' : ''}>
                {r.cells.map((c, j) => (
                  <td key={j}>{c}</td>
                ))}
              </tr>
              {r.chip && (
                <tr className="row-chip-row">
                  <td colSpan={data.headers.length}>
                    <div className="explain-chip row-chip">
                      <span className="chip-icon" aria-hidden="true">ℹ</span>
                      <span>{r.chip}</span>
                    </div>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>

      {data.callout && <div className="callout callout-large">{data.callout}</div>}

      {data.subTable && (
        <div className="subtable">
          <div className="subtable-label">{data.subTable.label}</div>
          <div className="subtable-cells">
            {data.subTable.cells.map((c, i) => (
              <div key={i} className="subtable-cell">
                <span className="subtable-k">{c.k}</span>
                <span className="subtable-v">{c.v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

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
