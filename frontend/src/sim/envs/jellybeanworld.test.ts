import { describe, expect, test } from 'bun:test';
import { lensRecording } from '../decode';
import { OFF_POSE, RECORD_FLOATS } from '../record';
import {
  jbwPolarity,
  jellyBeanWorldDecoder,
  JBW_HEADING,
  JBW_JELLIES,
  JBW_POLARITY_PERIOD,
  JBW_X,
  JBW_Y,
} from './jellybeanworld';

describe('jbwPolarity', () => {
  test('matches the env sign (step/100000 & 1) with the row-to-env-step shift', () => {
    expect(jbwPolarity(-1)).toBe(1);
    expect(jbwPolarity(0)).toBe(1);
    expect(jbwPolarity(JBW_POLARITY_PERIOD - 2)).toBe(1);
    expect(jbwPolarity(JBW_POLARITY_PERIOD - 1)).toBe(-1);
    expect(jbwPolarity(2 * JBW_POLARITY_PERIOD - 2)).toBe(-1);
    expect(jbwPolarity(2 * JBW_POLARITY_PERIOD - 1)).toBe(1);
  });
});

describe('collection log', () => {
  test('stamps the collected cell from the post-step row', () => {
    const steps = 4;
    const data = new Float32Array(steps * RECORD_FLOATS);
    const row = (i: number, x: number, y: number, heading: number, jellies: number): void => {
      const at = i * RECORD_FLOATS + OFF_POSE;
      data[at + JBW_X] = x;
      data[at + JBW_Y] = y;
      data[at + JBW_HEADING] = heading;
      data[at + JBW_JELLIES] = jellies;
    };
    row(0, 0, -1, 0, 0);
    row(1, 0, -2, 0, 0);
    row(2, 1, -2, 1, 1);
    row(3, 2, -2, 1, 1);
    const lens = lensRecording(
      { envId: 'JellyBeanWorld-cont-v0', seed: 7, data, steps },
      jellyBeanWorldDecoder,
    );
    const log = lens.events;
    expect(log.count).toBe(1);
    expect(log.types[log.kinds[0] ?? -1]).toBe('collection');
    expect(log.steps[0]).toBe(2);
    expect(log.cellX[0]).toBe(1);
    expect(log.cellY[0]).toBe(-2);
    expect(lens.events).toBe(log);
    const events = lens.eventsAt(3);
    const first = events[0];
    expect(first?.type).toBe('collection');
    if (first?.type === 'collection') {
      expect(first.x).toBe(1);
      expect(first.y).toBe(-2);
      expect(first.reward).toBe(1);
    }
  });
});
