import { DELTREL_RULES_CONTRACT } from './rules';

/** Display notation evolves independently of game, model, and saved-move identity. */
export const DELTREL_NOTATION_SCHEMA_ID = 'deltrel.board-notation.v3' as const;
export const DELTREL_NOTATION_VERSION = 3 as const;
export const COORDINATE_ARMS = ['A', 'B', 'C', 'D', 'E'] as const;
export const DELTREL_NOTATION_CONTRACT = {
  schema: DELTREL_NOTATION_SCHEMA_ID,
  version: DELTREL_NOTATION_VERSION,
  sectors: COORDINATE_ARMS,
  sectorOrder: 'clockwise, beginning at the lower-right radial arm',
  ring: 'distance from the center, 1 through 9; 0 denotes ring 10',
  position: 'clockwise steps from the named arm, 0 through ring minus 1',
  format: 'sector letter, ring digit, position digit',
} as const;

function assertSupportedRings(rings: number): void {
  if (!DELTREL_RULES_CONTRACT.board.supportedRings.includes(rings as 4 | 6 | 8 | 10)) {
    throw new Error('coordinates require a supported board size');
  }
}

/** A three-symbol polar address, such as A30 (on arm A) or A32 (two steps along). */
export function coordinateAt(
  rings: number,
  sector: number,
  ring: number,
  position: number,
): string {
  assertSupportedRings(rings);
  if (
    !Number.isInteger(sector) || sector < 0 || sector > 4 ||
    !Number.isInteger(ring) || ring < 1 || ring > rings ||
    !Number.isInteger(position) || position < 0 || position >= ring
  ) throw new Error('invalid board address for coordinate');
  return `${COORDINATE_ARMS[sector]}${ring % 10}${position}`;
}

/** Resolve a stable numeric move id into the selected board's display notation. */
export function coordinateLabel(nodeId: number, rings: number): string {
  assertSupportedRings(rings);
  if (!Number.isSafeInteger(nodeId) || nodeId < 0 || nodeId >= 5 * rings * (rings + 1) / 2) {
    throw new Error('node id must be an in-range non-negative safe integer');
  }
  let ring = 1;
  while (nodeId >= 5 * ring * (ring + 1) / 2) ring++;
  const offset = nodeId - 5 * ring * (ring - 1) / 2;
  return coordinateAt(rings, Math.floor(offset / ring), ring, offset % ring);
}
