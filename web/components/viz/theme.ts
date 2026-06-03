export const ACCENTS = {
  hero: '#3fb950',
  villain: '#6e7681',
  third: '#a371f7',
  fourth: '#d29922',
  bar: '#388bfd',
  grid: '#30363d',
  text: '#8b949e',
};

export const PLAYER_COLORS = [ACCENTS.hero, ACCENTS.third, ACCENTS.fourth, '#db61a2', '#1f6feb'];

export const CATEGORY_LABELS: Record<string, string> = {
  highcard: 'High', pair: 'Pair', twopair: '2 Pair', trips: 'Trips', straight: 'Straight',
  flush: 'Flush', fullhouse: 'Boat', quads: 'Quads', straightflush: 'St.Flush',
};
export const CATEGORY_ORDER = ['highcard', 'pair', 'twopair', 'trips', 'straight', 'flush', 'fullhouse', 'quads', 'straightflush'];
