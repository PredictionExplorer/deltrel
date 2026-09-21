import { DELTREL_RULES_CONTRACT } from './rules';

/** Display notation evolves independently of game, model, and saved-move identity. */
export const DELTREL_NOTATION_SCHEMA_ID = 'deltrel.board-notation.v2' as const;
export const DELTREL_NOTATION_VERSION = 2 as const;
export const DELTREL_NOTATION_CONTRACT = {
  schema: DELTREL_NOTATION_SCHEMA_ID,
  version: DELTREL_NOTATION_VERSION,
  files: 'letters left to right',
  ranks: 'positive integers bottom to top',
  cellUnits: 800_000_000,
  vertexX: [587_785_252, -587_785_252, -951_056_516, 0, 951_056_516],
  vertexYUp: [-809_016_994, -809_016_994, 309_016_994, 1_000_000_000, 309_016_994],
  rounding: 'nearest integer; exact halves away from zero',
  origin: 'minimum rounded cape column and rank on the selected board',
} as const;

export interface CoordinateGrid {
  readonly schema: typeof DELTREL_NOTATION_SCHEMA_ID;
  readonly version: typeof DELTREL_NOTATION_VERSION;
  readonly rings: number;
  /** Cell width/height in the original normalized screen geometry. */
  readonly step: number;
  readonly minColumn: number;
  readonly minRow: number;
  readonly columns: readonly { readonly label: string; readonly x: number }[];
  readonly ranks: readonly { readonly label: string; readonly y: number }[];
}

function roundCell(value: number): number {
  const cell = DELTREL_NOTATION_CONTRACT.cellUnits;
  return Math.sign(value) * Math.floor((Math.abs(value) + cell / 2) / cell);
}

export function createCoordinateGrid(rings: number): CoordinateGrid {
  if (!DELTREL_RULES_CONTRACT.board.supportedRings.includes(rings as 4 | 6 | 8 | 10)) {
    throw new Error('coordinate grid requires a supported board size');
  }
  const minColumn = roundCell(-951_056_516 * rings);
  const minRow = roundCell(-809_016_994 * rings);
  const maxRow = roundCell(1_000_000_000 * rings);
  const step = 0.8 / rings;
  return {
    schema: DELTREL_NOTATION_SCHEMA_ID,
    version: DELTREL_NOTATION_VERSION,
    rings,
    step,
    minColumn,
    minRow,
    columns: Array.from({ length: -2 * minColumn + 1 }, (_, column) => ({
      label: String.fromCharCode(65 + column),
      x: (column + minColumn) * step,
    })),
    ranks: Array.from({ length: maxRow - minRow + 1 }, (_, rank) => ({
      label: String(rank + 1),
      y: -(rank + minRow) * step,
    })),
  };
}

/** Integer arithmetic makes the spatial grid identical in every runtime. */
export function coordinateAt(
  grid: CoordinateGrid,
  sector: number,
  ring: number,
  position: number,
): { column: number; rank: number; label: string } {
  if (
    !Number.isInteger(sector) || sector < 0 || sector > 4 ||
    !Number.isInteger(ring) || ring < 1 || ring > grid.rings ||
    !Number.isInteger(position) || position < 0 || position >= ring
  ) throw new Error('invalid board address for coordinate');
  const next = (sector + 1) % 5;
  const { vertexX, vertexYUp } = DELTREL_NOTATION_CONTRACT;
  const x = (ring - position) * vertexX[sector] + position * vertexX[next];
  const y = (ring - position) * vertexYUp[sector] + position * vertexYUp[next];
  const column = roundCell(x) - grid.minColumn;
  const rank = roundCell(y) - grid.minRow + 1;
  return { column, rank, label: `${grid.columns[column].label}${rank}` };
}

/** Resolve a stable numeric move id into the selected board's display notation. */
export function coordinateLabel(nodeId: number, rings: number): string {
  const grid = createCoordinateGrid(rings);
  if (!Number.isSafeInteger(nodeId) || nodeId < 0 || nodeId >= 5 * rings * (rings + 1) / 2) {
    throw new Error('node id must be an in-range non-negative safe integer');
  }
  let ring = 1;
  while (nodeId >= 5 * ring * (ring + 1) / 2) ring++;
  const offset = nodeId - 5 * ring * (ring - 1) / 2;
  return coordinateAt(grid, Math.floor(offset / ring), ring, offset % ring).label;
}
