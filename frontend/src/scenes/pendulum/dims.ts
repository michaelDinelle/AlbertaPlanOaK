import { PD_L } from '@sim/envs/pendulum';

export const ROD_LEN = PD_L;

export const LANTERN_R = 0.135;
export const LANTERN_H = 0.23;
export const CAP_R = 0.108;
export const RIB_INSET = 0.99;
export const RIB_HALF = 0.0065;
export const TASSEL_CORD_LEN = 0.085;
export const TASSEL_WEIGHT_H = 0.085;
export const TASSEL_WEIGHT_R = 0.03;
export const TASSEL_DROP = 0.16;
export const FINIAL_H = 0.055;
export const RING_R = 0.022;
export const RING_TUBE = 0.006;

export const LANTERN_REACH = LANTERN_H * 0.5 + TASSEL_DROP + TASSEL_WEIGHT_H * 0.5;

export const LANTERN_HALF_Z = LANTERN_R;

export const SWING_REACH = ROD_LEN + LANTERN_REACH;

export const STRAP_HALF = 0.008;
export const STRAP_Z = LANTERN_HALF_Z + STRAP_HALF + 0.014;
export const HANGER_PLATE_H = 0.03;
export const HANGER_PLATE_W = 0.24;
export const HANGER_PLATE_DROP = 0.062;
export const SWING_CLEARANCE = 0.055;

export const PIN_R = 0.019;
export const EYE_TUBE = 0.011;
export const EYE_DROP = 0.026;
export const EYE_MAJOR = EYE_DROP + PIN_R + EYE_TUBE + 0.004;
