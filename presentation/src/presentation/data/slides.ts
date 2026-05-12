// src/presentation/data/slides.ts
export type SlideKind =
  | 'title'
  | 'content'
  | 'table'
  | 'dual'
  | 'conclusion'
  | 'chart'
  | 'image'
  | 'thanks';

export type ChartId = 'rmse' | 'walkforward' | 'losscurve';

export interface BaseSlide {
  kind: SlideKind;
  title?: string;
  subtitle?: string;
  chartId?: ChartId;
  /** Inline explanation chips rendered below main content, above bridge. */
  chips?: string[];
  /** Single italic narrative bridge line at the very bottom of the slide. */
  bridge?: string;
}

export interface ImageItem {
  src: string;
  alt: string;
  caption?: string;
  chip?: string;
}
export interface ImageSlide extends BaseSlide {
  kind: 'image';
  title: string;
  subtitle?: string;
  images: ImageItem[];
  badges?: { label: string; value: string }[];
  footnote?: string;
}

export interface ChartSlide extends BaseSlide {
  kind: 'chart';
  title: string;
  subtitle?: string;
  chartId: ChartId;
  caption?: string;
}

export interface TitleSlide extends BaseSlide {
  kind: 'title';
  mainTitle: string;
  subtitle: string;
  tagline: string;
  context: string;
  narrativeContext?: string;
}

export interface ContentBullet {
  text: string;
  highlight?: boolean;
  chip?: string;
}
export interface ContentSlide extends BaseSlide {
  kind: 'content';
  title: string;
  subtitle?: string;
  bullets: ContentBullet[];
  callout?: string;
  footnote?: string;
}

export interface TableRow {
  cells: string[];
  winner?: boolean;
  chip?: string;
}
export interface TableSlide extends BaseSlide {
  kind: 'table';
  title: string;
  subtitle?: string;
  headers: string[];
  rows: TableRow[];
  callout?: string;
  subTable?: {
    label: string;
    cells: { k: string; v: string }[];
  };
  footnote?: string;
}

export interface DualColumnContent {
  heading: string;
  bullets: string[];
  tone?: 'positive' | 'negative';
  chip?: string;
}
export interface DualSlide extends BaseSlide {
  kind: 'dual';
  title: string;
  subtitle?: string;
  left: DualColumnContent;
  right: DualColumnContent;
}

export interface ConclusionSlide extends BaseSlide {
  kind: 'conclusion';
  title: string;
  items: { tag: string; text: string }[];
  finalNote?: string;
  summaryChip?: string;
}

export interface ThanksSlide extends BaseSlide {
  kind: 'thanks';
  eyebrow?: string;
  headline: string;
  subline?: string;
  context?: string;
}

export type Slide =
  | TitleSlide
  | ContentSlide
  | TableSlide
  | DualSlide
  | ConclusionSlide
  | ChartSlide
  | ImageSlide
  | ThanksSlide;

export const slides: Slide[] = [
  // --- Slide 1 — Title ---
  {
    kind: 'title',
    mainTitle:
      'Automated Valuation Models for London Residential Real Estate (1995–2024)',
    subtitle: 'Trees vs. Neural Networks under Temporal Shift',
    tagline: '418,201 transactions · 29 years · 4 model families',
    context: 'Neural Networks Course Project · FABIZ-ASE · 2026',
    narrativeContext:
      'Why this project? London prices doubled relative to income over 29 years — can a model predict what a property is worth today, trained only on the past?',
  },

  // --- Slide 2 — The Problem ---
  {
    kind: 'content',
    title: 'The Problem & Why It Matters',
    subtitle: 'Motivation',
    bullets: [
      {
        text: 'London — the most volatile residential market in the UK, with direct macro-economic impact',
      },
      {
        text: 'Price-to-income: ~4× (1995) → >10× (2006) → GFC → COVID "space race" → high rates 2022–2024',
      },
      {
        text: 'AVMs (Automated Valuation Models) = scalable, objective, real-time valuation',
        highlight: true,
      },
      {
        text: 'Central challenge: data leakage — saleEstimate_* / rentEstimate_* columns excluded from training (policy decision)',
        chip:
          'These columns contain the vendor’s own price estimate — using them as features would let the model see the answer. Excluded by policy.',
      },
      {
        text: 'Target: history_price (continuous GBP, heavy-tailed log-normal distribution)',
      },
    ],
    bridge: 'Next: what does the data actually look like, and how do we split it fairly?',
  },

  // --- Slide 3 — EDA Overview ---
  {
    kind: 'image',
    title: 'Dataset Quality — EDA Overview',
    subtitle: 'Missingness profile and target distribution before any modeling',
    images: [
      {
        src: '/results/eda_missing.png',
        alt: 'Top 20 columns by missing percentage',
        caption: 'Top 20 features by missing % — vendor estimate columns dominate',
        chip:
          'Vendor columns (saleEstimate_*, rentEstimate_*) are the most missing — and also excluded from mainline features. Both reasons point to the same decision.',
      },
      {
        src: '/results/eda_target_dist.png',
        alt: 'Histogram of log1p(history_price)',
        caption: 'log1p(history_price) distribution — near-Gaussian after transform',
        chip:
          'log1p(price) is near-Gaussian → justifies PyTorch’s log-target transformation on Slide 12.',
      },
    ],
    bridge: 'Next: how we split 418k rows so the model never sees the future.',
  },

  // --- Slide 4 — Dataset & Temporal Split ---
  {
    kind: 'table',
    title: 'Dataset & Temporal Split',
    subtitle: 'Calendar-elapsed split, not row-count',
    headers: ['Set', 'Period', 'Rows (approx)'],
    rows: [
      { cells: ['Train', '1995-01-02 → 2015-03-04 (70% calendar)', '~185k'] },
      { cells: ['Validation', '2015-03-04 → 2018-04-13 (10% calendar)', '~53k'] },
      { cells: ['Test', '2018-04-13 → 2024-09-27 (20% calendar)', '~130k'], winner: true },
    ],
    callout:
      '418,201 raw rows → 102,527 exact duplicates removed → quantile clipping (1%–99%) selected on validation',
    footnote:
      'All preprocessing (imputation, scaling, OHE) fit exclusively on train — applied blind to val/test.',
    chips: [
      'Splitting by calendar time (not row index) prevents recent years — which have far more transactions — from leaking into the training window.',
      'If the scaler saw test data, it would know future price ranges. This is a form of data leakage.',
    ],
    bridge: 'Next: which model families competed on this split?',
  },

  // --- Slide 5 — Model Families ---
  {
    kind: 'table',
    title: 'Model Families Compared',
    subtitle: 'Same preprocessor · same split · same evaluation matrix',
    headers: ['Family', 'Architecture'],
    rows: [
      {
        cells: ['Baseline', 'NaiveMedian (zero-intelligence reference)'],
        chip:
          'NaiveMedian always predicts the training median (~£270k). It is the zero-intelligence floor — any real model must beat it.',
      },
      { cells: ['Linear', 'Ridge, ElasticNet'] },
      { cells: ['Tree Ensemble', 'HistGradientBoostingRegressor, RandomForestRegressor'] },
      { cells: ['NN sklearn', 'MLPRegressor (small / medium / large capacity scan)'] },
      {
        cells: ['NN PyTorch', 'MLPNet custom (dropout 0.2, AdamW, SmoothL1, log-target + bias init)'],
        chip:
          'Course-required model: explicit training loop, architecture diagram, and loss curves on Slides 12–13.',
      },
    ],
    callout: 'Same evaluation matrix across all models — absolute fairness',
    bridge: 'Next: who won on the held-out test period 2018–2024?',
  },

  // --- Slide 6 — Main Results ---
  {
    kind: 'table',
    title: 'Main Results — Static Holdout (Test 2018–2024)',
    subtitle: 'The impact slide',
    headers: ['Model', 'Test RMSE (£)', 'Test MAE (£)', 'R²'],
    rows: [
      { cells: ['HistGBR ✓', '393,489', '216,083', '0.720'], winner: true },
      { cells: ['RandomForest', '396,947', '215,149', '~0.715'] },
      { cells: ['Ridge', '562,968', '—', '—'] },
      { cells: ['MLP_medium (sklearn)', '645,104', '—', '—'] },
      { cells: ['TorchMLP (PyTorch)', '729,943', '—', '—'] },
      { cells: ['NaiveMedian', '927,646', '561,899', '—'] },
    ],
    callout: 'HistGBR: 57.6% RMSE reduction vs NaiveMedian',
    subTable: {
      label: '5-quantile bin classification (non-technical proxy)',
      cells: [
        { k: 'Price-bin Accuracy', v: '49.51%' },
        { k: 'Macro F1', v: '0.495' },
      ],
    },
    chips: [
      'R² = 0.720 means the model explains 72% of the variance in London house prices on unseen data from 2018–2024.',
      'Accuracy of 49.51% on 5 price bands ≈ almost 2× a random classifier (20%) — useful for stakeholders who think in bands, not £ errors.',
    ],
    bridge: 'Next: why do trees systematically beat neural networks on this type of data?',
  },

  // --- Slide 7 — Regression Diagnostics ---
  {
    kind: 'image',
    title: 'Regression Diagnostics — HistGBR Test Set',
    subtitle: 'Where the winning model is accurate, and where it breaks',
    images: [
      {
        src: '/results/regression_pred_vs_actual.png',
        alt: 'Predicted vs actual prices, HistGBR test set',
        caption: 'Predicted vs Actual · R² = 0.720 · Test 2018–2024',
        chip:
          'Points on the dashed diagonal = perfect prediction. The spread above £2M shows where the model underestimates luxury properties.',
      },
      {
        src: '/results/regression_residuals.png',
        alt: 'Residuals vs predicted values',
        caption: 'Residuals vs Predicted · heavy tail visible above £1.5M',
        chip:
          'Residuals cluster tightly near zero but fan out above £1.5M — the heavy tail the model cannot fully capture. This drives the segment gate FAIL on Detached and Semi-Detached.',
      },
    ],
  },

  // --- Slide 8 — 5-Bin Classification ---
  {
    kind: 'image',
    title: '5-Bin Classification View — Non-Technical Accuracy Proxy',
    subtitle: 'Continuous regression translated into discrete price bands for stakeholders',
    images: [
      {
        src: '/results/price_bin_confusion.png',
        alt: '5-bin price confusion matrix',
        caption: 'Heaviest errors stay on diagonal ± 1 bin',
      },
    ],
    badges: [
      { label: 'Accuracy', value: '49.51%' },
      { label: 'Macro F1', value: '0.495' },
    ],
    chips: [
      'Each row = true price band (cheapest=0, most expensive=4). Each column = predicted band. Diagonal = correct. Off-diagonal errors are mostly ±1 bin — the model is rarely wrong by more than one price tier.',
    ],
  },

  // --- Slide 9 — RMSE Comparison Chart ---
  {
    kind: 'chart',
    title: 'RMSE Comparison — All Models',
    subtitle: 'Test set 2018–2024 · sorted ascending · NaiveMedian = zero-intelligence baseline',
    chartId: 'rmse',
    caption:
      'HistGBR wins by a thin margin over RandomForest; all NN variants sit above linear models on this static holdout.',
    chips: [
      'Only £3,458 RMSE difference between HistGBR and RandomForest — essentially tied on the static holdout. The walk-forward tells a different story.',
      'All NN variants sit 60–120% above the tree winners on this single holdout. Calendar shift (train on 1995–2015, test on 2018–2024 at much higher prices) disadvantages NNs more than boosting.',
    ],
    bridge: 'Next: the literature explains why trees win here — and where NNs recover.',
  },

  // --- Slide 10 — Why Trees Win ---
  {
    kind: 'content',
    title: 'Why Trees Win on Static Holdout',
    subtitle: 'Literature confirms our findings — Grinsztajn et al. 2022 · Shwartz-Ziv & Armon 2022',
    bullets: [
      {
        text: 'Robustness to uninformative features — tree splits ignore noise columns; NNs do not',
        highlight: true,
        chip:
          'Our dataset has many sparse OHE columns and several near-constant features. Trees ignore them; NNs propagate noise through all layers.',
      },
      {
        text: 'Discontinuous boundaries — property prices have hard edges (postcode, catchment areas, taxes) → NN spectral bias hurts',
        highlight: true,
        chip:
          'A postcode boundary can cause a £200k price jump in 100 metres. Trees model this as a split; NNs approximate it with smooth activation functions.',
      },
      {
        text: 'No scaling requirements — trees are invariant to monotone transforms; NNs need precise preprocessing',
        highlight: true,
      },
    ],
    callout:
      'On a single-shot holdout with large calendar shift (1995→2024), boosting clearly beats NNs',
    bridge:
      'But wait — a single holdout is not the whole story. What happens when we test across multiple time windows?',
  },

  // --- Slide 11 — Walk-Forward ---
  {
    kind: 'table',
    title: 'Walk-Forward Validation — Where NNs Fight Back',
    subtitle: 'Protocol: 4 calendar folds, expanding window (2000→2006→2012→2018→2024)',
    headers: ['Metric', 'RandomForest', 'MLP (sklearn)'],
    rows: [
      { cells: ['Mean RMSE', '347,500 £', '356,658 £'] },
      { cells: ['RMSE Std', '84,775', '53,909'] },
      {
        cells: ['CV%', '24.4%', '15.1% ✓'],
        winner: true,
        chip:
          'CV% = RMSE standard deviation / mean RMSE. Lower = more consistent across different macro regimes. MLP CV is 38% lower than RF — it adapts better as the market shifts.',
      },
    ],
    callout: 'MLP beats RF in 2 of 4 folds · CV 38% lower → more temporally stable',
    footnote: 'In real production, where macro regimes shift, the NN is the more robust candidate.',
    chartId: 'walkforward',
    chips: [
      'During 2006–2012 (pre/post GFC) and 2012–2018 (recovery), the MLP generalises better. These are the periods with the largest structural breaks.',
    ],
    bridge: 'Next: how was the PyTorch MLP built to handle heavy-tailed price data without exploding?',
  },

  // --- Slide 12 — PyTorch MLP Safeguards ---
  {
    kind: 'content',
    title: 'PyTorch MLP — Architectural Safeguards',
    subtitle: 'Three critical design choices for heavy-tailed financial data',
    bullets: [
      {
        text: 'Log-target transformation — log1p(y) → near-Gaussian → gradient stability',
        highlight: true,
        chip:
          'Without this, the model trains on £300k–£5M values with high variance. After log1p, the target range is ~12–15, much easier for gradient descent.',
      },
      {
        text: 'Strategic bias initialization — final_layer.bias = mean(log1p(y_train)) → eliminates Initial Guessing Bias',
        highlight: true,
        chip:
          'Without this, random initialisation can produce log(price) ≈ 24, meaning expm1(24) ≈ £26 billion per property. Observed empirically on the first run.',
      },
      {
        text: 'Log-clipping — clamp(pred_log, min_log − margin, max_log + margin) → impossible to predict £26B prices',
        highlight: true,
        chip:
          'Safety net: even if training diverges slightly, predictions stay within the observed price envelope ± a small margin.',
      },
      {
        text: 'Architecture: dropout(0.2) · AdamW · SmoothL1Loss · early stopping on val RMSE in GBP space',
      },
    ],
    chartId: 'losscurve',
  },

  // --- Slide 13 — Training Convergence ---
  {
    kind: 'image',
    title: 'PyTorch MLP — Training Convergence',
    subtitle: 'Pipeline-generated loss curves: PyTorch vs sklearn',
    images: [
      {
        src: '/results/torch_mlp_loss_curve.png',
        alt: 'PyTorch MLP validation RMSE per epoch',
        caption: 'TorchMLP · Best checkpoint epoch 21 · Val RMSE 428,185 £',
        chip:
          'Val RMSE stops decreasing consistently after epoch 21 — early stopping restores that checkpoint. Training loss still falls, showing overfitting would occur without stopping.',
      },
      {
        src: '/results/MLP_medium_loss_curve.png',
        alt: 'sklearn MLP_medium loss curve',
        caption: 'sklearn MLP_medium · same split · Val RMSE plateau visible',
        chip:
          'sklearn MLP uses StandardScaler on the target (not log), so early stopping is honest but the model cannot exploit the log-normal structure. This is why TorchMLP has a dedicated training loop.',
      },
    ],
  },

  // --- Slide 14 — Dual-Track & Limitations ---
  {
    kind: 'dual',
    title: 'Dual-Track & Honest Limitations',
    subtitle: 'What works with vendor features vs. where the model fails',
    left: {
      heading: 'Assisted Track (with vendor features)',
      tone: 'positive',
      bullets: [
        'AssistedHistGBR Test RMSE: 304,436 £ (vs 393,489 mainline)',
        'External benchmark saleEstimate_lowerPrice: 350,168 £',
        'But: depends on vendor data → not independently deployment-safe',
      ],
      chip:
        'AssistedHistGBR RMSE (304,436 £) even beats the vendor’s own estimate (350,168 £) — the model adds value on top of the vendor signal. But this requires vendor data at inference time.',
    },
    right: {
      heading: 'Honest Limitations',
      tone: 'negative',
      bullets: [
        'Segment gate FAIL on: Detached, Semi-Detached, price_band > 1M £',
        'MAPE = 33.8% → within-10% rate only 24.3% (below commercial AVM standard)',
        'Large calendar shift: train 1995–2015 vs test 2018–2024 (much higher prices)',
      ],
      chip:
        'Semi-Detached RMSE = 825,628 £ vs overall 393,489 £ — more than 2× worse. These segments fail the automated release gate and would need targeted retraining before production.',
    },
  },

  // --- Slide 15 — Conclusions ---
  {
    kind: 'conclusion',
    title: 'Conclusions & Key Takeaways',
    summaryChip:
      'The full story in one sentence: a gradient boosting model trained on 20 years of London transactions predicts 2018–2024 prices with 57.6% lower error than a naive baseline, while neural networks — despite losing on this single holdout — prove more temporally stable across rolling windows.',
    items: [
      {
        tag: 'RQ1',
        text: 'HistGBR dominates static holdout — R² 0.720, 57.6% RMSE uplift vs naive',
      },
      {
        tag: 'RQ2',
        text: 'MLP is 38% more stable on walk-forward (CV 15.1% vs 24.4%) → better production candidate under temporal drift',
      },
      {
        tag: 'RQ3',
        text: 'Bias initialization + log-clipping are critical for PyTorch MLP convergence on heavy-tailed financial data',
      },
      {
        tag: 'RQ4',
        text: '5-bin classification proxy (accuracy 49.51%) translates regression into accessible language for non-technical stakeholders',
      },
      {
        tag: '⊕',
        text: 'No universal winner — trees for peak static accuracy, NNs for temporal stability in production',
      },
    ],
  },

  // --- Slide 16 — Thank You ---
  {
    kind: 'thanks',
    eyebrow: 'End of Presentation',
    headline: 'Thank you',
    subline: 'Questions?',
    context: 'London Housing AVM · FABIZ-ASE · 2026',
  },
];
