// src/presentation/charts/index.tsx
import type { ChartId } from '../data/slides';
import { RMSEBarChart } from './RMSEBarChart';
import { WalkForwardChart } from './WalkForwardChart';
import { LossCurveChart } from './LossCurveChart';

export function ChartById({ id }: { id: ChartId }) {
  switch (id) {
    case 'rmse':
      return <RMSEBarChart />;
    case 'walkforward':
      return <WalkForwardChart />;
    case 'losscurve':
      return <LossCurveChart />;
  }
}
