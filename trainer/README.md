# PLO Trainer question banks

Offline generator for the training modes at train.magicflop.com
(repo `plotrainer_frontend`). Answers are computed here once and shipped as
static JSON; nothing in this folder runs in the engine server.

| Mode | Question | Answer |
| --- | --- | --- |
| `range_reads` | How often does {pos}'s opening range have X on this flop? | exact weighted combo count |
| `card_removal` | Same, but you hold a hand (its cards are removed from villain's range) | exact, plus the no-blocker baseline |
| `range_vs_range` | Which of two opening ranges has X more often? | exact, both values |
| `hero_equity` | Your hand's equity vs {pos}'s whole opening range | Monte Carlo, 40k runouts (about ±0.5 pts) |

X is one of: flush draw, nut flush draw, set, top set, flopped straight, two
pair or better, flopped flush, 8+ straight outs, 13+ out wrap, combo draw, the
nuts, air. Definitions live in `features.py` and are shown to players.

## Ranges

- **PLO4**: GTO raise-first-in ranges, PLO50 100BB 6-max (UTG/MP/CO/BTN/SB),
  from the trainer's DynamoDB solver data; weight = raise frequency.
  Stored in `data/gto_rfi_plo50_100bb.npz` (cards, weights, BTN raise EV).
- **PLO5** (no solves exist, so assumed): same opening share per position as
  PLO4; hands ranked by the average PLO4 strength (BTN raise-EV percentile) of
  their five 4-card sub-hands, with trips/quads discounted the way the PLO4
  solve discounts them. See `ranges.py`.

## Regenerate

```bash
python -m trainer.build --flops 60 \
  --out ../plotrainer_frontend/public/data \
  --out web/data/trainer
```

Both copies must match: the trainer serves practice questions from its own
copy, and `web/` scores challenges against its copy (so scores can't be faked
from the browser). Takes ~25 minutes, mostly PLO5.

Tests: `python -m pytest tests/test_trainer.py` (features vs hand-worked
cases and the PQL engine, range sizes, mode invariants, equity vs engine).
