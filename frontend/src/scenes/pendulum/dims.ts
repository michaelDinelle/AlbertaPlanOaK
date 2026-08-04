// The Lantern Bell — the swept envelope of the hanging lantern, in ONE place.
//
// WHY THIS FILE EXISTS. The torii has to be built around the volume the lantern actually
// sweeps, and the lantern is built from the numbers below. When those two lived in
// separate files the arch was sized against a guess (`ROD_LEN + 0.19`) that was 0.13
// short of the truth, and at |θ| ≳ 2.8 — which includes the fleet init at θ = π — the
// tassel weight sat INSIDE the tie beam and the hanger cables ran through the lantern's
// paper barrel. So the dimensions live here, the reach is DERIVED from them, `actors.ts`
// builds the lantern from them, `world.ts` places every beam from the reach, and the test
// measures the built meshes and asserts the derivation still describes them.
//
// Everything is in world units, and 1 world unit is 1 env unit: the swing circle really
// is the env's own r = PD_L.

import { PD_L } from '@sim/envs/pendulum';

/** The env's own rod length. Every dimension in the scene hangs off this. */
export const ROD_LEN = PD_L;

// ---- the lantern, in the BOB's own frame (local +y is "up the rod") ----------

/** paper-barrel radius at the waist — also the assembly's half-thickness in ±z */
export const LANTERN_R = 0.135;
/** barrel height between the caps */
export const LANTERN_H = 0.23;
/** cap / shoulder radius */
export const CAP_R = 0.108;
/** the six corner ribs sit on the barrel at this fraction of LANTERN_R */
export const RIB_INSET = 0.99;
export const RIB_HALF = 0.0065;
/** the tassel: a short cord and a weight, the silhouette that says "hanging" */
export const TASSEL_CORD_LEN = 0.085;
export const TASSEL_WEIGHT_H = 0.085;
export const TASSEL_WEIGHT_R = 0.03;
/** the weight's centre below the barrel's bottom rim */
export const TASSEL_DROP = 0.16;
/** finial + hanging ring, above the top cap */
export const FINIAL_H = 0.055;
export const RING_R = 0.022;
export const RING_TUBE = 0.006;

/**
 * The farthest any lantern vertex sits from the BOB. The tassel weight's apex is the far
 * point, and because the whole assembly only ever ROTATES about the bob (the hook's give
 * is a rotation too), this radius is what the lantern sweeps, at every θ and every lag.
 */
export const LANTERN_REACH = LANTERN_H * 0.5 + TASSEL_DROP + TASSEL_WEIGHT_H * 0.5;

/**
 * Half-thickness across the swing plane. The barrel's hexagon is the widest part of the
 * assembly (the ribs sit inside it at RIB_INSET·R + RIB_HALF < R), and rotation about z
 * never changes z — so this is invariant in θ.
 */
export const LANTERN_HALF_Z = LANTERN_R;

// ---- what the assembly sweeps about the PIVOT --------------------------------

/** outer radius of the swept annulus — nothing static may sit inside it, in the slab */
export const SWING_REACH = ROD_LEN + LANTERN_REACH;
/** inner radius: the lantern never comes closer to the pivot than this (the pin is safe) */
export const SWING_INNER = ROD_LEN - LANTERN_REACH;

// ---- the hanger ---------------------------------------------------------------

/** the cable cross-section (square) */
export const STRAP_HALF = 0.008;
/**
 * The cables stand off the swing plane far enough that the paper barrel passes BETWEEN
 * them at the top of the arc: the near face of a cable is clear of the barrel's widest
 * point, with air to spare. This is the constraint the shipped 0.105 broke — it was
 * INSIDE a 0.135 barrel, so the cables cut a dark line across the lit paper at θ ≈ π.
 */
export const STRAP_Z = LANTERN_HALF_Z + STRAP_HALF + 0.014;
/** the plate that bolts the cables to the underside of the tie beam */
export const HANGER_PLATE_H = 0.03;
export const HANGER_PLATE_W = 0.24;
/** the plate's mid-plane below the beam's mid-plane */
export const HANGER_PLATE_DROP = 0.062;
/** air between the tassel's apex at the top of the arc and the lowest ironwork */
export const SWING_CLEARANCE = 0.055;

// ---- the pivot hardware: the rod's eye rides on the torii's pin ---------------
//
// These two are built in different files and have to agree, so they live here. An eyelet
// hangs on a pin with the pin inside its HOLE — which means the hole must clear the pin
// measured from the far side of the drop, not from the eye's own centre. The first pass
// used a 0.034 ring dropped 0.035 below a 0.019 pin, so the pin's axis fell outside the
// hole entirely and the pin cut straight through the ring's wire at every θ.

export const PIN_R = 0.019;
export const EYE_TUBE = 0.011;
/** the eye hangs this far below the pivot, as a forged eyelet on a pin does */
export const EYE_DROP = 0.026;
/** hole radius = EYE_MAJOR − EYE_TUBE, and it has to reach past the drop AND the pin */
export const EYE_MAJOR = EYE_DROP + PIN_R + EYE_TUBE + 0.004;
