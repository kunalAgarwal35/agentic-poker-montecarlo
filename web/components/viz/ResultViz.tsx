'use client';
import type { VizSpec } from '@/lib/types';
import { EquityChart } from '@/components/viz/EquityChart';
import { WinTieLossChart } from '@/components/viz/WinTieLossChart';
import { DistributionChart } from '@/components/viz/DistributionChart';
import { EquityByStreetChart } from '@/components/viz/EquityByStreetChart';
import { EquityDistributionChart } from '@/components/viz/EquityDistributionChart';
import { EquityVsClassChart } from '@/components/viz/EquityVsClassChart';
import { FrequencyStat } from '@/components/viz/FrequencyStat';
import { DrawsChart } from '@/components/viz/DrawsChart';
import { OutsDistributionChart } from '@/components/viz/OutsDistributionChart';

export function ResultViz({ spec }: { spec: VizSpec }) {
  switch (spec.kind) {
    case 'equity': return <EquityChart rows={spec.rows} />;
    case 'win-tie-loss': return <WinTieLossChart rows={spec.rows} />;
    case 'distribution': return <DistributionChart player={spec.player} bars={spec.bars} />;
    case 'equity-by-street': return <EquityByStreetChart series={spec.series} />;
    case 'equity-distribution':
      return <EquityDistributionChart player={spec.player} mean={spec.mean} combos={spec.combos} sampledCombos={spec.sampledCombos} buckets={spec.buckets} />;
    case 'equity-vs-class':
      return <EquityVsClassChart player={spec.player} rows={spec.rows} />;
    case 'frequency': return <FrequencyStat label={spec.label} pct={spec.pct} />;
    case 'draws': return <DrawsChart player={spec.player} bars={spec.bars} />;
    case 'outs-distribution':
      return <OutsDistributionChart player={spec.player} handtype={spec.handtype} street={spec.street} avg={spec.avg} bars={spec.bars} />;
  }
}
