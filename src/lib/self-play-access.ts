import { createHash, timingSafeEqual } from 'node:crypto';

// Only a digest is published. The randomly generated link is kept outside Git.
const SELF_PLAY_SECRET_SHA256 = 'bfb3f46937b12479960a08016c20671d01920e4d349db95ca3972769d2d928f7';

/** Only the server validates the private link; clients receive a boolean. */
export function hasSelfPlayAccess(value: unknown, secret: string | undefined): boolean {
  if (typeof value !== 'string' || value.length < 32 || value.length > 512) return false;
  if (secret !== undefined && secret.length < 32) return false;
  const supplied = createHash('sha256').update(value).digest();
  const expected = secret === undefined
    ? Buffer.from(SELF_PLAY_SECRET_SHA256, 'hex')
    : createHash('sha256').update(secret).digest();
  return timingSafeEqual(supplied, expected);
}
