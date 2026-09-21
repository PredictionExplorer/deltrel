/** Player palette shared by the board and the panels. */
export const PLAYER_COLORS = [
  {
    name: 'Clay',
    base: '#ed9b76',
    bright: '#ffd9b8',
    deep: '#9a4b39',
    glow: 'rgba(237, 155, 118, 0.55)',
    soft: 'rgba(237, 155, 118, 0.16)',
  },
  {
    name: 'Pearl',
    base: '#f1e8d6',
    bright: '#fff9ef',
    deep: '#9c8a70',
    glow: 'rgba(241, 232, 214, 0.55)',
    soft: 'rgba(241, 232, 214, 0.16)',
  },
] as const;

export const BOARD_PRESETS = [
  { rings: 4, label: 'Mini', nodes: 50, shores: 20 },
  { rings: 6, label: 'Small', nodes: 105, shores: 30 },
  { rings: 8, label: 'Medium', nodes: 180, shores: 40 },
  { rings: 10, label: 'Full', nodes: 275, shores: 50 },
] as const;
