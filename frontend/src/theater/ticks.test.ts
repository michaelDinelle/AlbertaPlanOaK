import { describe, expect, test } from 'bun:test';
import { SIM_EVENT_TYPES } from '@sim/types';
import { TICK_KINDS } from './loader';

describe('timeline ticks', () => {
  test('every sim event kind an environment can emit appears on the tape', () => {
    for (const type of SIM_EVENT_TYPES) {
      expect(TICK_KINDS).toContain(type);
    }
  });
});
