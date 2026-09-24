#include "../env.h"

typedef struct { float target; } calibrationKnobs;

static const calibrationKnobs CALIBRATION_KNOBS[2] = {
    { 0.3f },
    { 0.8f }
};


void calibration_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) { (void)steps; (void)instance; (void)seed; (void)words; (void)extra; }

float calibration_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra;
    const calibrationKnobs *k = &CALIBRATION_KNOBS[instance];
    *done = 0u;
    float frequency = wordFloat(words[0]), toward = ((uint32_t)ctrl[0] & 1u) ? 1.0f : 0.0f;
    float rate = frequency >= 0x1p-120f ? 0x1p-6f : toward * 0x1p-6f;
    frequency = fmaf(toward - frequency, rate, frequency);
    words[0] = floatWord(frequency);
    float miss = frequency - k->target;
    obs[0] = frequency;
    obs[1] = 1.0f;
    return -miss * miss;
}

void calibration_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
    (void)shared;
    (void)instance; (void)seed; (void)extra; (void)step;
    obs[0] = wordFloat(words[0]);
    obs[1] = 1.0f;
}

void calibration_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
    (void)shared;
    (void)seed; (void)step; (void)extra; (void)obs;
    ctrl[0] = wordFloat(words[0]) < (&CALIBRATION_KNOBS[instance])->target ? 1.0f : 0.0f;
}
