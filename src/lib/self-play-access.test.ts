import { describe, expect, it } from 'vitest';
import { hasSelfPlayAccess } from './self-play-access';

const secret = 'a-long-private-self-play-secret-for-tests';

describe('server self-play access', () => {
  it('requires the exact configured secret and rejects missing, guessed, and repeated query values', () => {
    expect(hasSelfPlayAccess(secret, secret)).toBe(true);
    for (const value of [undefined, '', 'true', '1', secret + 'x', secret.toUpperCase(), [secret], [secret, secret]]) {
      expect(hasSelfPlayAccess(value, secret)).toBe(false);
    }
    expect(hasSelfPlayAccess(secret, undefined)).toBe(false);
    expect(hasSelfPlayAccess('short', 'short')).toBe(false);
  });
});
