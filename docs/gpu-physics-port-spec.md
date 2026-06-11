# MuJoCo 3.6.0 per-step computation spec — HalfCheetah (planar), for Metal fp32 port

**Source of truth:** github.com/google-deepmind/mujoco, tag `3.6.0` (tag exists; all line
numbers below are against that tag). Files read:
`src/engine/engine_forward.c`, `engine_core_smooth.c`, `engine_core_constraint.c`,
`engine_solver.c`, `engine_collision_driver.c`, `engine_collision_primitive.c`,
`engine_passive.c`, `engine_support.c`, `engine_core_util.c`, `engine_util_spatial.c`,
`engine_util_solve.c`, `engine_util_blas.c`, `engine_util_sparse.c`, `engine_island.c`,
`engine_init.c`, `engine_io.c`, `engine_util_misc.c`, `include/mujoco/mjmodel.h`.
Model: Gymnasium `half_cheetah.xml` (the classic cheetah model, also used by envpool).

This document is a **contract**: every formula and loop below is what MuJoCo 3.6.0
executes for this exact model, with dead code paths removed. "MODEL FACT" lines are
properties of this model that justify removing a branch.

---

## 0. Global constants, model facts, and data layout

### 0.1 Engine constants (mujoco.h / mjmodel.h)

| constant | value | used in |
|---|---|---|
| `mjMINVAL` | `1e-15` | division floors everywhere |
| `mjMAXVAL` | `1e10` | `mju_isBad` checks |
| `mjMINMU`  | `1e-5` | friction floor (`mj_assignFriction`) |
| `mjMINIMP` | `0.0001` | impedance clamp low |
| `mjMAXIMP` | `0.9999` | impedance clamp high |
| `mjMAXCONPAIR` | 50 | max contacts per geom pair (we use ≤ 2) |

### 0.2 mjOption for this model (engine_init.c:51–106 defaults + XML `<option>`)

```
timestep    = 0.01            gravity   = (0, 0, -9.81)
integrator  = mjINT_EULER     cone      = mjCONE_PYRAMIDAL
solver      = mjSOL_NEWTON    jacobian  = mjJAC_AUTO  → DENSE (nv=9 < 60)
iterations  = 100             tolerance = 1e-8
ls_iterations = 50            ls_tolerance = 0.01
noslip_iterations = 0         impratio  = 1
disableflags = 0              enableflags = 0
```

Consequences (verify in `mj_isSparse`, engine_core_util.c:32–39):
- **Dense constraint Jacobian** (`nv = 9 < 60`), dense nv×nv Newton Hessian/Cholesky.
- **Pyramidal cone** ⇒ elliptic-cone code paths never execute.
- `noslip_iterations = 0` ⇒ **`mj_solNoSlip` is never called** (engine_forward.c:819–821)
  and `mj_isDual()==0` ⇒ **`mj_projectConstraint` (efc_AR) is a no-op**
  (engine_core_constraint.c:2512+, early return for non-dual solvers).
- `disableflags == 0` ⇒ EULERDAMP **enabled** (implicit joint damping in Euler),
  WARMSTART **enabled**, REFSAFE **enabled**, FILTERPARENT **enabled**,
  **ISLAND discovery enabled** (`mjDSBL_ISLAND` is a *disable* bit, mjmodel.h:71;
  default 0 ⇒ islands ON). See §6.0 for why this is benign here.
- `enableflags == 0` ⇒ no energy, no fwdinv, no override, no sleep, no multiCCD.
  All `sleep_filter` branches below are FALSE; they are omitted from the pseudocode.

### 0.3 Model facts (from half_cheetah.xml + compiler defaults)

Bodies (`nbody = 8`): `0=world, 1=torso, 2=bthigh, 3=bshin, 4=bfoot, 5=fthigh, 6=fshin, 7=ffoot`.
`body_parentid = [0, 0, 1, 2, 3, 1, 5, 6]`. `body_rootid[i] = 1` for all i ≥ 1.
`body_weldid[i] = i` (every non-world body has joints).

Joints (`njnt = nq = nv = 9`), one dof per joint, `jnt_qposadr[j] = jnt_dofadr[j] = j`:

| j | name | type | axis (local) | body | armature | damping | stiffness | limited | range |
|---|------|------|------|------|---|---|---|---|---|
| 0 | rootx | slide | (1,0,0) | 1 | 0 | 0 | 0 | no | — |
| 1 | rootz | slide | (0,0,1) | 1 | 0 | 0 | 0 | no | — |
| 2 | rooty | hinge | (0,1,0) | 1 | 0 | 0 | 0 | no | — |
| 3 | bthigh | hinge | (0,1,0) | 2 | 0.1 | 6 | 240 | **yes** | [−.52, 1.05] |
| 4 | bshin | hinge | (0,1,0) | 3 | 0.1 | 4.5 | 180 | **yes** | [−.785, .785] |
| 5 | bfoot | hinge | (0,1,0) | 4 | 0.1 | 3 | 120 | **yes** | [−.4, .785] |
| 6 | fthigh | hinge | (0,1,0) | 5 | 0.1 | 4.5 | 180 | **yes** | [−1, .7] |
| 7 | fshin | hinge | (0,1,0) | 6 | 0.1 | 3 | 120 | **yes** | [−1.2, .87] |
| 8 | ffoot | hinge | (0,1,0) | 7 | 0.1 | 1.5 | 60 | **yes** | [−.5, .5] |

All `jnt_pos = (0,0,0)` (joints anchored at body frame origin).
`jnt_margin = 0` for all. `qpos0 = 0` (9 zeros) and `qpos_spring = 0`
(**springref default is 0, not qpos0** — `mj_springdamper` uses `m->qpos_spring`,
engine_passive.c:172; here both are 0 so it is moot, but the reference array is
`qpos_spring`).
Joint-limit solver params (from `<joint solreflimit=".02 1" solimplimit="0 .8 .03">`,
remaining solimp entries default):
`jnt_solref = [0.02, 1]`, `jnt_solimp = [0, 0.8, 0.03, 0.5, 2]`.

**Joint limits ARE constraints in this model.** `limited="true"` on the 6 leg joints ⇒
`mjCNSTR_LIMIT_JOINT` rows are added whenever a joint is at/past its range (margin 0,
§2.6.1). The 3 root joints have `limited="false"` ⇒ never produce rows.

Geoms (`ngeom = 9`):

| g | name | type | body | size | contype | conaffinity |
|---|------|------|------|------|---|---|
| 0 | floor | plane | 0 | (40,40,40) | 1 | 1 |
| 1 | torso | capsule | 1 | r=0.046, hl=0.5 (fromto, axis +x) | 1 | 0 |
| 2 | head | capsule | 1 | r=0.046, hl=0.15 | 1 | 0 |
| 3 | bthigh | capsule | 2 | r=0.046, hl=0.145 | 1 | 0 |
| 4 | bshin | capsule | 3 | r=0.046, hl=0.15 | 1 | 0 |
| 5 | bfoot | capsule | 4 | r=0.046, hl=0.094 | 1 | 0 |
| 6 | fthigh | capsule | 5 | r=0.046, hl=0.133 | 1 | 0 |
| 7 | fshin | capsule | 6 | r=0.046, hl=0.106 | 1 | 0 |
| 8 | ffoot | capsule | 7 | r=0.046, hl=0.07 | 1 | 0 |

All geoms: `condim=3`, `friction=(0.4, 0.1, 0.1)`, `solref=[0.02,1]`,
`solimp=[0, 0.8, 0.01, 0.5, 2]`, `margin=0`, `gap=0`, `priority=0`, `solmix=1`.

**Collision pair set is static** (§2.5.1): exactly the 8 pairs
`(floor, capsule_g)` for g = 1..8, in that order. Capsule–capsule pairs are
bitmask-filtered (1&0, 1&0). Each plane–capsule pair yields **0, 1, or 2**
contact points (one per capsule end sphere). `ncon ≤ 16`.

Actuators (`nu = 6`), all `mjTRN_JOINT` motors with fixed gain 1, no bias, no act state
(`na = 0`):

| u | joint (dof) | gear | ctrlrange |
|---|---|---|---|
| 0 | bthigh (3) | 120 | [−1,1] |
| 1 | bshin (4) | 90 | [−1,1] |
| 2 | bfoot (5) | 60 | [−1,1] |
| 3 | fthigh (6) | 120 | [−1,1] |
| 4 | fshin (7) | 60 | [−1,1] |
| 5 | ffoot (8) | 30 | [−1,1] |

No tendons, no equality constraints, no dof frictionloss (`efc_frictionloss = 0`
always, `nf = 0`), no sensors, no flexes, no mocap, no plugins, no user callbacks.
`ne = 0` always. So **efc rows = (joint limit rows) then (contact rows)**, in that order.

Compile-time constants treated as inputs to the kernel (copy them out of a compiled
mjModel once, with `settotalmass="14"` and `inertiafromgeom="true"` applied):
`body_mass[8]`, `body_inertia[8][3]`, `body_ipos[8][3]`, `body_iquat[8][4]`,
`body_pos[8][3]`, `body_quat[8][4]` (= identity for all bodies in this model),
`body_subtreemass[8]`, `body_invweight0[8][2]`, `dof_invweight0[9]`,
`dof_armature[9]`, `dof_damping[9]`, `jnt_stiffness[9]`, `geom_pos/quat/size`,
`geom_sameframe`, `body_sameframe`, `stat.meaninertia` (not needed on the island path).

### 0.4 DOF tree and sparse inertia layout (CSR, "M" format)

`dof_parentid = [-1, 0, 1, 2, 3, 4, 2, 6, 7]`
(rootx←nothing, rootz←rootx, rooty←rootz, bthigh←rooty, bshin←bthigh, bfoot←bshin,
fthigh←rooty, fshin←fthigh, ffoot←fshin).
`dof_bodyid = [1,1,1,2,3,4,5,6,7]`. `dof_simplenum = 0` for all 9 dofs.

`d->M`, `d->qLD`, `d->qH` all use the same CSR layout of the **lower triangle**
(row i = ancestors of i in increasing dof order, diagonal last):

```
row 0: cols [0]                       rowadr  0, rownnz 1
row 1: cols [0,1]                     rowadr  1, rownnz 2
row 2: cols [0,1,2]                   rowadr  3, rownnz 3
row 3: cols [0,1,2,3]                 rowadr  6, rownnz 4
row 4: cols [0,1,2,3,4]               rowadr 10, rownnz 5
row 5: cols [0,1,2,3,4,5]             rowadr 15, rownnz 6
row 6: cols [0,1,2,6]                 rowadr 21, rownnz 4
row 7: cols [0,1,2,6,7]               rowadr 25, rownnz 5
row 8: cols [0,1,2,6,7,8]             rowadr 30, rownnz 6
total nC = nM = 36 nonzeros
```

`M_rownnz = [1,2,3,4,5,6,4,5,6]`, `M_rowadr = [0,1,3,6,10,15,21,25,30]`,
`M_colind` as listed. Diagonal address of row i: `M_rowadr[i] + M_rownnz[i] - 1`.
(d->qM is a legacy-layout copy made by `mju_scatter(qM, M, mapM2M)` in `mj_makeM`,
engine_core_smooth.c:1816–1822; nothing in the per-step pipeline below reads qM, so
the port can skip it.)

### 0.5 Per-step state

Persistent across steps: `qpos[9]`, `qvel[9]`, `qacc_warmstart[9]`, `time`.
Inputs each step: `ctrl[6]` (also `qfrc_applied[9]`, `xfrc_applied[8][6]`, both zero in
envpool/gym usage — the port may hard-wire them to zero; the spec keeps qfrc_applied in
§5 for completeness).
Everything else is recomputed every step.

---

## 1. `mj_step` — top-level call order (engine_forward.c:1493–1527)

```
mj_step(m, d):
  mj_checkPos(m, d)                       # engine_forward.c:54
  mj_checkVel(m, d)                       # engine_forward.c:72
  mj_forward(m, d)                        #   = mj_forwardSkip(stage NONE, skipsensor=0)
  mj_checkAcc(m, d)                       # engine_forward.c:93
  # ENBL_FWDINV off → no mj_compareFwdInv
  mj_Euler(m, d)                          # integrator == mjINT_EULER
```

`mj_forwardSkip` (engine_forward.c:1436–1483) with no sensors and no energy enabled:

```
mj_forwardSkip:
  mj_fwdPosition(m, d)                    # §2
  # mj_sensorPos: nsensor==0 → no-op
  # energy disabled → d->energy = {0,0}
  mj_fwdVelocity(m, d)                    # §3
  # mj_sensorVel: no-op
  # mjcb_control == NULL
  mj_fwdActuation(m, d)                   # §4
  mj_fwdAcceleration(m, d)                # §5
  mj_fwdConstraint(m, d)                  # §6
  d->flg_rnepost = 0
  # mj_sensorAcc: nsensor==0 → no-op  (⇒ mj_rnePostConstraint NEVER RUNS, §8)
```

Sanity checks (exact):

```
mju_isBad(x) := (x != x) || (x > 1e10) || (x < -1e10)      # engine_util_misc.c:1608

mj_checkPos: if any of qpos[0..8] isBad → mj_resetData(m,d) (§9), warning count++, return.
mj_checkVel: same over qvel[0..8].
mj_checkAcc: same over qacc[0..8]; on trigger: mj_resetData THEN mj_forward(m,d)
             (recompute the whole forward pass from the reset state).
```

`mj_step1/mj_step2` are not used by `mj_step`. Sensors/energy are skipped exactly as
shown (the lazy flags `flg_energypos/flg_subtreevel/flg_rnepost` never get consumed).

---

## 2. `mj_fwdPosition` (engine_forward.c:153–218)

```
mj_fwdPosition:
  mj_kinematics(m, d)        # §2.1   (mj_fwdKinematics also calls mj_comPos, camlight, flex, tendon)
  mj_comPos(m, d)            # §2.2
  # mj_camlight: 1 camera, writes cam_xpos/cam_xmat only — physics-irrelevant, SKIP
  # mj_flex, mj_tendon: nflex=ntendon=0 → no-op
  mj_makeM(m, d)             # §2.3 = mj_crb + (tendon armature no-op) + scatter to qM
  mj_factorM(m, d)           # §2.4
  mj_collision(m, d)         # §2.5
  mj_makeConstraint(m, d)    # §2.6
  mj_island(m, d)            # §2.7 (1 island when nefc>0; identity maps)
  mj_transmission(m, d)      # §2.8
  # mj_projectConstraint: not dual → no-op
```

### 2.1 `mj_kinematics` (engine_core_smooth.c:40–242)

World body: `xpos[0]=0`, `xquat[0]=(1,0,0,0)`, `xmat[0]=I`, `xipos[0]=0`, `ximat[0]=I`.

For body i = 1..7 (kinematics1, engine_core_smooth.c:55–186). No free joints, no mocap.
MODEL FACT: all `body_quat[i] = (1,0,0,0)`; geometry orientation lives in `geom_quat`
(from `axisangle`/`fromto` attributes) and `body_iquat`, both handled in kinematics2.

```
for i in 1..7:
    pid = body_parentid[i]
    # parent frame compose (body_pos, body_quat=identity)
    if pid == 0:  xpos = body_pos[i];                     xquat = body_quat[i]      # torso
    else:         xpos = xmat[pid] @ body_pos[i] + xpos[pid]
                  xquat = quatmul(xquat[pid], body_quat[i])                          # = xquat[pid]
    for each joint j on body i (torso: j = 0,1,2 in order; legs: single hinge):
        xaxis_j   = rotVecQuat(jnt_axis[j], xquat)        # see note below
        xanchor_j = rotVecQuat(jnt_pos[j], xquat) + xpos  # jnt_pos = 0 → xanchor_j = xpos
        if jnt_type[j] == SLIDE:
            xpos += xaxis_j * (qpos[j] - qpos0[j])        # qpos0 = 0
        else:  # HINGE
            qloc  = axisAngle2Quat(jnt_axis[j], qpos[j] - qpos0[j])
                    # = (cos(a/2), axis*sin(a/2)); angle==0 → (1,0,0,0)  [spatial.c:96]
            xquat = quatmul(xquat, qloc)
            xpos  = xanchor_j - rotVecQuat(jnt_pos[j], xquat)   # jnt_pos = 0 → xpos = xanchor_j
        store xanchor[j], xaxis[j]
    normalize4(xquat)        # floor: if |q| < 1e-15 → (1,0,0,0); skip divide if ||q|-1| ≤ 1e-15
    store xpos[i], xquat[i]; xmat[i] = quat2Mat(xquat[i])
```

MODEL FACT (useful invariant, follows from the above): every xquat is a pure y-rotation,
so `xaxis` is **constant**: rootx → (1,0,0), rootz → (0,0,1), all 7 hinges → (0,1,0).
`xanchor` for torso joints = torso `xpos` (before/after slide offsets as per loop order:
rootx anchor = frame origin before x-slide is applied, etc. — if you exploit the
invariant, keep the exact accumulation order above; xanchor of joint j is the body
origin *after* the preceding joints' transforms).

kinematics2 (engine_core_smooth.c:191–232) — inertial frames and geoms via
`mj_local2Global` (engine_core_util.c:848–889):

```
for i in 1..7:   # body inertial frame; sameframe shortcut per compiled flag
    xipos[i] = xmat[i] @ body_ipos[i] + xpos[i]
    ximat[i] = quat2Mat( quatmul(xquat[i], body_iquat[i]) )      # (or copies if sameframe)
for g in 0..8:   # geoms (g=0 on world: xmat=I → geom_xpos=geom_pos, geom_xmat=quat2Mat(geom_quat))
    geom_xpos[g] = xmat[b] @ geom_pos[g] + xpos[b]
    geom_xmat[g] = quat2Mat( quatmul(xquat[b], geom_quat[g]) )
```

quaternion formulas: `mju_mulQuat` engine_util_spatial.c:66–77, `mju_quat2Mat` :145–190
(standard; quat2Mat has an exact-identity shortcut `q==(1,0,0,0) → I`).

fp32 notes: `normalize4` divides only if `| ||q|| − 1 | > 1e-15`; in fp32 replace this
with an unconditional normalize (the conditional is a double-precision micro-opt) — it
changes bits vs double MuJoCo regardless.

### 2.2 `mj_comPos` (engine_core_smooth.c:246–350) — subtree COM, cinert, cdof

```
# subtree center of mass
for i in 0..7: subtree_com[i] = xipos[i] * body_mass[i]          # moment init
for i in 7..1 (backward): subtree_com[parent(i)] += subtree_com[i]
for i in 0..7:
    if body_subtreemass[i] < 1e-15: subtree_com[i] = xipos[i]
    else: subtree_com[i] /= body_subtreemass[i]

# spatial inertia of each body in the frame centered at its subtree-root COM
cinert[0] = 0 (10 zeros)
for i in 1..7:
    off = xipos[i] - subtree_com[body_rootid[i]]      # rootid = 1 → subtree_com[1] = robot COM
    cinert[i] = inertCom(body_inertia[i], ximat[i], off, body_mass[i])
```

`mju_inertCom` (engine_util_spatial.c:401–431), output packing
`cinert = [Ixx Iyy Izz Ixy Ixz Iyz, mass*off_x mass*off_y mass*off_z, mass]` (10 numbers):

```
tmp = diag(inert) @ ximatᵀ      # 3x3, rows scaled
rot = ximat @ tmp               # only 6 unique entries computed:
  c0=rot00, c1=rot11, c2=rot22, c3=rot01, c4=rot02, c5=rot12
c0 += m*(off1²+off2²);  c1 += m*(off0²+off2²);  c2 += m*(off0²+off1²)
c3 -= m*off0*off1;      c4 -= m*off0*off2;      c5 -= m*off1*off2
c6 = m*off0;  c7 = m*off1;  c8 = m*off2;  c9 = m
```

```
# motion dofs in the same frame (cdof[j] is 6: [angular(3); linear(3)])
for each joint j on body i:
    off = subtree_com[body_rootid[i]] - xanchor[j]     # = com_robot - anchor
    SLIDE  (j=0,1):  cdof[j] = [0,0,0, xaxis_j]                      # dofCom(axis, NULL)
    HINGE  (j=2..8): cdof[j] = [xaxis_j, cross(xaxis_j, off)]        # dofCom(axis, off)
```

(`mju_dofCom` engine_util_spatial.c:446–458.)

### 2.3 `mj_crb` → `M` (engine_core_smooth.c:1740–1813)

```
crb[0..7] = cinert[0..7]                       # copy 10 numbers per body
for i in 7..1 (backward):                      # accumulate composite inertia
    if body_parentid[i] > 0: crb[parent(i)] += crb[i]   (10 adds)

M[0..35] = 0
for i in 0..8:                                 # dof_simplenum all 0 → full path
    adr_diag = M_rowadr[i] + M_rownnz[i] - 1
    M[adr_diag] = dof_armature[i]
    buf6 = mulInertVec(crb[dof_bodyid[i]], cdof[i])     # spatial I * motion vector
    a = adr_diag
    j = i
    while j >= 0:
        M[a] += dot6(cdof[j], buf6);  a -= 1;  j = dof_parentid[j]
```

`mju_mulInertVec` (engine_util_spatial.c:435–442), with `i = cinert` packing above and
`v = [w(3); v(3)]`:

```
res[0] = i0*v0 + i3*v1 + i4*v2 - i8*v4 + i7*v5
res[1] = i3*v0 + i1*v1 + i5*v2 + i8*v3 - i6*v5
res[2] = i4*v0 + i5*v1 + i2*v2 - i7*v3 + i6*v4
res[3] = i8*v1 - i7*v2 + i9*v3
res[4] = i6*v2 - i8*v0 + i9*v4
res[5] = i7*v0 - i6*v1 + i9*v5
```

(`mj_tendonArmature` no-op; `qM` scatter skippable, §0.4.)

### 2.4 `mj_factorM` — sparse L'·D·L (engine_core_smooth.c:1889–1943)

`qLD = M` (copy 36), then `mj_factorI(qLD, qLDiagInv, 9, rownnz, rowadr, colind, NULL)`:

```
# In-place LDL of the CSR lower triangle. After this, row i holds
# L(i, ancestors)/D-normalized off-diagonals and D(i,i) at the diagonal slot.
for j in 8..0 (backward):
    start = rowadr[j];  diag = rownnz[j]-1;  end = start + diag
    invD = 1 / qLD[end]                       # NOTE: no mjMINVAL guard in this path!
    qLDiagInv[j] = invD
    for adr in end-1 .. start (backward):     # update ancestor rows
        i = colind[adr]
        # row i (length rownnz[i], same leading column pattern) -= row_j_prefix * (qLD[adr]*invD)
        qLD[rowadr[i] .. rowadr[i]+rownnz[i]-1] += qLD[start .. start+rownnz[i]-1] * (-qLD[adr]*invD)
    qLD[start .. start+diag-1] *= invD        # normalize off-diagonal of row j
```

This works because colind of row i is a **prefix-compatible** subset: every ancestor
row's column list is exactly the leading `rownnz[i]` entries of row j's list (tree
sparsity, §0.4). The fp32 port can rely on that.

fp32 note: the diagonal here has no floor (the legacy `mj_factorI_legacy` had a
`mjMINVAL` clamp; `mj_factorI` does not). M's diagonal is ≥ armature (0.1 for legs) but
root diagonals are pure mass terms ~14; fp32 is fine, but add a debug assert
`qLD[diag] > 0`.

Solve routine used everywhere (`mj_solveLD`, engine_core_smooth.c:2059–2142), n=1
vector, in-place `x ← (L'DL)⁻¹ x`:

```
# x ← L'⁻ᵀ ... precisely:  x ← L⁻ᵀ x  (rows backward, scatter to ancestors)
for i in 8..0:
    if rownnz[i] == 1: continue
    if x[i] != 0:
        for adr in rowadr[i] .. rowadr[i]+rownnz[i]-2:  x[colind[adr]] -= qLD[adr] * x[i]
# x ← D⁻¹ x
for i in 0..8: x[i] *= qLDiagInv[i]
# x ← L⁻¹ x  (rows forward, gather from ancestors)
for i in 0..8:
    if rownnz[i] == 1: continue
    x[i] -= Σ_{adr} qLD[adr] * x[colind[adr]]     for adr in row i off-diagonals
```

(The `if x[i] != 0` sparsity shortcut is bit-exact-neutral; keep or drop.)

### 2.5 Collision (engine_collision_driver.c)

#### 2.5.1 Broadphase — statically resolvable

`mj_broadphase` (driver:1249–1394): body 0 (world, has the plane and
`body_contype|conaffinity ≠ 0`) is "always-colliding": pairs (0,b) for every body b
whose `filterBodyPair(0,0,1, weldid_b, parentweld_b, 0, dsbl=0)` passes — for b=1..7
all pass (the parent filter `weldbody1 == weldparent2` is skipped because
`weldbody1 == 0`, driver:179). SAP may propose capsule-body pairs but
`canCollide2` (driver:201–210) filters all of them: `filterBitmask(1,0,1,0)` →
`!(1&0) && !(1&0)` = filtered.

⇒ **the broadphase always returns exactly the 7 body pairs (0,1)...(0,7), sorted by
signature** `(0<<16)+b`. The port should hard-code the geom pair list:

```
PAIRS = [(0,1),(0,2)] for body 1 (two geoms, inner loop g2 = 1,2)   # driver:457–462
      + [(0,3),(0,4),(0,5),(0,6),(0,7),(0,8)]
```

Contact records are appended in this pair order; each pair appends its (up to 2)
points in capsule-end order (end1 = +axis first). This ordering defines efc row order —
**must be preserved**.

#### 2.5.2 `mj_collideGeoms` per pair (driver:1548–1672)

```
g1 = plane(0), g2 = capsule (already type-ordered: PLANE(0) < CAPSULE(3))
# filterBitmask passes: contype[g2] & conaffinity[g1] = 1&1
margin = geom_margin[g1] + geom_margin[g2] = 0          # mj_assignMargin, no override
# mj_filterSphere (driver:139–156): plane case — skip pair if
#   planeGeomDist(plane, capsule_center) > margin + geom_rbound[capsule]
#   where planeGeomDist = dot(normal, xpos_capsule - xpos_plane); rbound = r + hl
num = mjc_PlaneCapsule(...)                              # §2.5.3, 0..2 contacts
if num == 0: continue
mj_contactParam(...)                                     # §2.5.4
for each contact: geom ids; mj_setContact(...)           # §2.5.5
ncon += num
```

#### 2.5.3 `mjc_PlaneCapsule` (engine_collision_primitive.c:62–88) — EXACT

```
mat1 = geom_xmat[plane] (= I for this model: plane at origin, identity orientation)
n    = (mat1[2], mat1[5], mat1[8])           # plane normal = third column = (0,0,1)
axis = (mat2[2], mat2[5], mat2[8])           # capsule local z in world
seg  = hl * axis                             # hl = geom_size[capsule][1]
r    = geom_size[capsule][0]                 # 0.046

for end in {+1, -1}:                         # END ORDER: +seg first, then −seg
    center = capsule_xpos + end*seg
    # mjraw_PlaneSphere (primitive.c:28–50):
    cdist = dot(center - plane_xpos, n)      # signed height of sphere center
    if cdist > margin + r: no contact for this end          # margin = 0
    else:
        dist  = cdist - r                                   # ≤ 0 given margin 0
        pos   = center - n * (dist/2 + r)                   # midpoint of penetration
        frame[0:3] = n
        frame[3:6] = axis                                   # set by PlaneCapsule (:80–85)
```

So a capsule resting exactly tangent (`cdist == r`) produces a contact record with
`dist == 0` which is then **excluded** (`exclude = dist >= includemargin = 0`,
§2.5.5) — it occupies a contact slot but adds no constraint rows.

#### 2.5.4 `mj_contactParam` (driver:1400–1493) — friction combination, VERIFIED

Both geoms have priority 0 → "same priority" branch:

```
condim = max(3, 3) = 3
mix    = solmix1/(solmix1+solmix2) = 0.5
solref = 0.5*[.02,1] + 0.5*[.02,1]  = [0.02, 1]        # both standard (ref0 > 0) → mix
solimp = 0.5*si + 0.5*si            = [0, 0.8, 0.01, 0.5, 2]
fri    = elementwise max( (0.4,.1,.1), (0.4,.1,.1) ) = (0.4, 0.1, 0.1)
# unpack to 5D contact friction:
friction = [fri[0], fri[0], fri[1], fri[2], fri[2]] = [0.4, 0.4, 0.1, 0.1, 0.1]
```

**Answer to the VERIFY:** the two tangential friction coefficients are both the
*slide* friction → `mu = (0.4, 0.4)` for condim 3. Torsional (0.1) and rolling (0.1)
entries exist in the 5-vector but are never read for condim 3.

#### 2.5.5 `mj_setContact` (driver:1497–1523)

```
con.dim = 3;  con.includemargin = margin - gap = 0
con.solref = [0.02,1]; con.solreffriction = [0,0]; con.solimp = [0,0.8,0.01,0.5,2]
con.friction = max(mjMINMU, ·) elementwise = [0.4,0.4,0.1,0.1,0.1]   # assignFriction
con.exclude = (con.dist >= 0)
mju_makeFrame(con.frame)        # engine_util_spatial.c:508–534:
    normalize3(frame[0:3])                                  # n already unit
    # frame[3:6] = capsule axis, |axis|=1 ≥ 0.5 → kept
    frame[3:6] -= n * dot(n, axis);  normalize3(frame[3:6]) # tangent1
    frame[6:9]  = cross(n, tangent1)                        # tangent2
con.efc_address = -1; con.mu = 0; con.H[36] = 0
```

fp32 note: when the capsule axis is nearly parallel to the normal
(|dot(n,axis)| → 1), tangent1 = axis − n·dot becomes tiny and `normalize3` floors at
1e-15 → direction noise. For cheetah capsules the axis stays well off vertical, but
guard anyway (this is the one genuinely ill-conditioned operation in collision).
If the y-axis slot were unset (sphere case — not here), makeFrame picks (0,1,0) or
(0,0,1); with PlaneCapsule the slot is always the capsule axis.

### 2.6 `mj_makeConstraint` (engine_core_constraint.c:2414–2508)

```
ne = nf = nl = nefc = 0
# (allocation precount — port: fixed max nefc = 12 limit rows? no: ≤ 6 limit rows
#  (one side active per joint; both sides only if range degenerate) + 16*4 contact rows = 70.
#  Safe static allocation: 6*2 + 16*4 = 76 rows × nv=9 dense J.)
mj_instantiateEquality: no-op (ne=0)
mj_instantiateFriction: no-op (nf=0)
mj_instantiateLimit     # §2.6.1 — appends nl rows
mj_instantiateContact   # §2.6.2 — appends 4 rows per active contact
mj_diagApprox           # §2.6.3
mj_makeImpedance        # §2.6.4
```

Each appended row sets (`mj_addConstraint`, constraint.c:258–349, dense branch):
`efc_J[row][0..8]` (dense 9-vector), `efc_pos`, `efc_margin`, `efc_frictionloss=0`,
`efc_type`, `efc_id`. Dense-mode empty-row check: limit rows always have a nonzero
entry; contact rows skip the check.

#### 2.6.1 Joint limits (`mj_instantiateLimit`, constraint.c:1082–1141)

```
for j in 3..8:                       # the 6 limited hinges, in joint order
    v = qpos[j]
    for side in {-1, +1}:            # lower first, then upper
        dist = side * (range[j][ (side+1)/2 ] - v)
               # side=-1: dist = v - range_lo ;  side=+1: dist = range_hi - v
        if dist < margin_j (= 0):    # i.e. STRICTLY past the limit
            J_row = zeros(9);  J_row[j] = -side
            append row: type=mjCNSTR_LIMIT_JOINT, id=j, pos=dist, margin=0
            nl += 1
```

NOTE: with margin = 0 a limit row exists **only when the joint is strictly beyond its
range** (dist < 0). Both sides can never be active simultaneously (ranges are wide).
Rows appear in joint order; they precede all contact rows.

#### 2.6.2 Contacts (`mj_instantiateContact`, constraint.c:1316–1413, pyramidal)

```
for i in 0..ncon-1 (contact record order from §2.5.1):
    if contact[i].exclude: continue
    con.efc_address = nefc
    # Jacobian difference: bodies (0=plane side, b=capsule body)  [mj_contactJacobian :1233]
    # dense: mj_jacDifPair → mj_jac twice (engine_core_util.c:176–224):
    jac1p = jac_point(world body 0)   = 0  (body_weldid[0]=0 → all-zero)
    jac2p = jac_point(con.pos, body b):
        offset = con.pos - subtree_com[1]
        for dof i walking b's ancestor chain (last dof of b, then dof_parentid links):
            jacp[:,i] = cdof[i][3:6] + cross(cdof[i][0:3], offset)
    jacdifp = jac2p - jac1p           # 3 x 9, rows = world xyz
    jacC = con.frame (3x3, rows = n, t1, t2) @ jacdifp      # 3 x 9, contact-frame rows
    # 4 pyramidal facet rows, in this exact order (k = tangent index 1,2):
    for k in 1..2:
        mu_k = con.friction[k-1]                  # = 0.4 both
        row(+): J = jacC[0] + mu_k * jacC[k]
        row(−): J = jacC[0] - mu_k * jacC[k]
        both rows: pos = con.dist, margin = con.includemargin (= 0),
                   type = mjCNSTR_CONTACT_PYRAMIDAL, id = i
    nefc += 4
```

`efc` layout summary: rows `[0..nl)` = limits, rows `[nl..nl+4*nactive_con)` = contacts,
4 consecutive rows per contact in facet order `(+t1, −t1, +t2, −t2)`.

#### 2.6.3 `mj_diagApprox` (constraint.c:1419–1633)

```
for limit rows:     dA[row] = dof_invweight0[ jnt_dofadr[id] ]     # = dof_invweight0[id]
for each contact (4 rows):
    tran = body_invweight0[2*0] + body_invweight0[2*b]    # world entry is 0 → = invweight of b
    for j in 0..1:                                        # facet pair index
        fri = con.friction[j]                             # 0.4
        dA[i+2j] = dA[i+2j+1] = tran + fri*fri*tran       # = tran * 1.16
```

#### 2.6.4 `mj_makeImpedance` (constraint.c:1811–1925) — KBIP, R, D

For every constraint *group* (limit row alone; contact = its first row, applied to all
4 rows; `getposdim` gives dim=1 for limit, dim = 2*(3−1) = 4 for pyramidal contact):

`getsolparam` (constraint.c:1637–1708):
```
limit:   solref=[0.02,1], solimp_raw=[0,0.8,0.03,0.5,2]
contact: solref=[0.02,1], solimp_raw=[0,0.8,0.01,0.5,2], solreffriction=[0,0]
REFSAFE: solref[0] = max(solref[0], 2*timestep) = max(0.02, 0.02) = 0.02   (no change)
clamp:   solimp[0] = clip(.,1e-4,.9999) = 1e-4    # the 0 gets clamped!
         solimp[1] = 0.8;  solimp[2] = width;  solimp[3] = 0.5;  solimp[4] = 2
```

`getimpedance` (constraint.c:1759–1807) with `pos = efc_pos[i]`, `margin = efc_margin[i] = 0`:
```
# d0=solimp[0]=1e-4, d1=0.8, width=solimp[2], mid=0.5, p=2;  d0 != d1, width > 1e-15
x = (pos - margin) / width ;  sgn = sign(x);  x = |x|
if x >= 1: imp = d1 ; impP = 0           # deep violation ( |pos| ≥ width )
elif x <= 0: imp = d0 ; impP = 0
else:
    if x <= mid:  a = 1/mid          (=2);  y = a*x²;     yP = 2*a*x
    else:         b = 1/(1-mid)      (=2);  y = 1-b*(1-x)²; yP = 2*b*(1-x)
    imp  = d0 + y*(d1-d0)
    impP = yP * sgn * (d1-d0) / width    # impP is unused downstream here (KBIP[3])
```
(With p = 2 the `power()` helper takes its exact `b==2 → a*a` branch, constraint.c:1748.)

Then per row j of the group (constraint.c:1828–1867):
```
R[i+j]  = max(1e-15, (1-imp) * dA[i+j] / imp)
K       = 1 / max(1e-15, solimp[1]² * solref[0]² * solref[1]²)
        = 1 / (0.64 * 4e-4 * 1) = 3906.25                       # both limits and contacts
B       = 2 / max(1e-15, solimp[1] * solref[0]) = 2/0.016 = 125
KBIP[4*(i+j)+0..3] = (K, B, imp, impP)
```
(Friction rows / elliptic-friction special cases don't occur: solreffriction = (0,0),
cone pyramidal, nf = 0.)

Contact friction-R adjustment (constraint.c:1874–1913), per contact (i = its first row):
```
R[i+1] = R[i] / max(1e-15, impratio=1) = R[i]
con.mu = friction[0] * sqrt(R[i+1]/R[i]) = 0.4
Rpy    = 2 * con.mu² * R[i] = 0.32 * R[i]
R[i+0..3] = Rpy                       # all four facet rows share one R
```
Finally:
```
efc_D[row] = 1 / R[row]               # for all rows
dA[row]    = R[row] * imp/(1-imp)     # diagApprox rewritten; unused downstream here
```

### 2.7 `mj_island` (engine_island.c:381–620)

Islands are ON by default in 3.6.0. MODEL FACT: one kinematic tree (`ntree = 1`,
all 9 dofs in tree 0) and every constraint touches tree 0 ⇒

- `nefc == 0` → `nisland = nidof = 0` (no island work).
- `nefc > 0` → exactly **1 island containing all 9 dofs and all rows**, and because
  dofs/rows are assigned in natural order (island.c:504–517, 570–575),
  `map_idof2dof = identity(9)`, `map_iefc2efc = identity(nefc)`,
  `iM = M`, `iLD = qLD`, `iLDiagInv = qLDiagInv`, `iefc_J = efc_J` (block-diag of one
  block), and all `island_*` arrays describe the single full-size block.

**Port consequence:** you may ignore islands entirely *except* for one numerical
detail — the Newton convergence `scale` differs on the island path (§6.4). Implement
the island value.

### 2.8 `mj_transmission` (engine_core_smooth.c:1212–1271, mjTRN_JOINT scalar case)

```
for u in 0..5:
    j = actuator_trnid[2u]                      # joint id (= dof id)
    actuator_length[u] = qpos[j] * gear[u]
    moment row u: single entry  moment[u][dof j] = gear[u]
```

---

## 3. `mj_fwdVelocity` (engine_forward.c:222–257)

```
# flexedge/tendon velocities: none
actuator_velocity[u] = gear[u] * qvel[dof_u]        # sparse moment row · qvel
mj_comVel        # §3.1
mj_passive       # §3.2
mj_referenceConstraint   # §3.3
mj_rne(flg_acc=0) → qfrc_bias    # §3.4
# mj_tendonBias: no-op
```

### 3.1 `mj_comVel` (engine_core_smooth.c:2205–2271)

```
cvel[0] = 0
for i in 1..7 (forward, parents before children — true for body order here):
    cvel = cvel[parent(i)]
    for each dof j on body i (torso: 0,1,2 in order; legs: 1 dof):
        # HINGE/SLIDE both take the default branch:
        cdof_dot[j] = crossMotion(cvel, cdof[j])       # uses cvel BEFORE adding dof j
        cvel       += cdof[j] * qvel[j]
    cvel[i] = cvel
```
`mju_crossMotion(res, v, m)` (engine_util_spatial.c:371–382), v=[w;l], m=[mw;ml]:
```
res[0:3] = cross(v[0:3], m[0:3])
res[3:6] = cross(v[0:3], m[3:6]) + cross(v[3:6], m[0:3])
```

### 3.2 `mj_passive` (engine_passive.c:632–720, springdamper :118–189)

```
qfrc_spring = qfrc_damper = qfrc_passive = 0   (gravcomp/fluid arrays too; unused)
# springs (stiffness > 0 only on dofs 3..8):
for j in 3..8: qfrc_spring[j] = -jnt_stiffness[j] * (qpos[j] - qpos_spring[j])   # qpos_spring = 0
# dampers (damping > 0 only on dofs 3..8):
for j in 3..8: qfrc_damper[j] = -dof_damping[j] * qvel[j]
qfrc_passive = qfrc_spring + qfrc_damper
# gravcomp: ngravcomp = 0 → skip;  fluid: density = viscosity = 0 → skip
# mj_contactPassive: only acts on contacts with exclude == 4 (flex "passive" contacts) → no-op here
```

### 3.3 `mj_referenceConstraint` (engine_core_constraint.c:2747–2759)

```
efc_vel = efc_J @ qvel                              # dense nefc x 9 matvec
for each row i:
    efc_aref[i] = -KBIP[4i+1]*efc_vel[i] - KBIP[4i+0]*KBIP[4i+2]*(efc_pos[i] - efc_margin[i])
               =  -B*vel - K*imp*(pos - margin)     # margin = 0 throughout this model
```

### 3.4 `mj_rne` with `flg_acc = 0` → `qfrc_bias` (engine_core_smooth.c:2354–2416)

```
cacc[world] = [0,0,0, -gravity] = [0,0,0, 0,0,9.81]
for i in 1..7 (forward):
    bda = body_dofadr[i]; n = body_dofnum[i]
    cacc[i] = cacc[parent(i)] + Σ_{k<n} cdof_dot[bda+k] * qvel[bda+k]
    # flg_acc == 0 → NO cdof*qacc term
    cfrc[i] = mulInertVec(cinert[i], cacc[i])
            + crossForce(cvel[i], mulInertVec(cinert[i], cvel[i]))
cfrc[world] = 0
for i in 7..1 (backward): if parent(i) != 0: cfrc[parent(i)] += cfrc[i]
for v in 0..8: qfrc_bias[v] = dot6(cdof[v], cfrc[dof_bodyid[v]])
```
`mju_crossForce(res, v, f)` (engine_util_spatial.c:386–397), v=[w;l], f=[t;f]:
```
res[0:3] = cross(v[0:3], f[0:3]) + cross(v[3:6], f[3:6])
res[3:6] = cross(v[0:3], f[3:6])
```

---

## 4. `mj_fwdActuation` (engine_forward.c:274–567), reduced to this model

```
ctrl_local[u] = clip(d->ctrl[u], -1, +1)      for u in 0..5     # CLAMPCTRL on, ctrllimited
if any ctrl_local isBad: ctrl_local = 0 (all six) + warning
# na = 0 → no act_dot;  gaintype FIXED gainprm[0] = 1, biastype NONE:
actuator_force[u] = 1 * ctrl_local[u]
# no force ranges, no tendons, no gravcomp actuators, no jnt_actfrcrange
qfrc_actuator = momentᵀ @ actuator_force:
    qfrc_actuator[0..2] = 0
    qfrc_actuator[dof_u] = gear[u] * ctrl_local[u]      # dofs 3..8
```

Note: `d->ctrl` itself is NOT clamped/zeroed — only the local copy.

---

## 5. `mj_fwdAcceleration` (engine_forward.c:571–604)

```
qfrc_smooth = qfrc_passive - qfrc_bias + qfrc_applied + qfrc_actuator   # qfrc_applied = 0 in envpool
# mj_xfrcAccumulate: xfrc_applied all-zero → memcmp shortcut, no-op
qacc_smooth = copy(qfrc_smooth)
mj_solveLD(qacc_smooth, qLD, qLDiagInv, 9, 1, ...)        # §2.4 solve, in place
```

---

## 6. `mj_fwdConstraint` → Newton solver

### 6.0 Driver (engine_forward.c:736–824)

```
qfrc_constraint = 0
if nefc == 0:
    qacc = qacc_smooth; solver_niter = 0; return        # airborne / inside limits
efc_b = efc_J @ qacc_smooth - efc_aref                  # nefc dense matvec
warmstart(m, d)                                          # §6.1
# islands_supported = TRUE (islands on, nisland=1, noslip 0, Newton)
# gather to i-arrays = identity copies (§2.7); then:
mj_solNewton_island(m, d, island=0, maxiter=100)        # = mj_solPrimal(island=0, flg_Newton=1)
# scatter back = identity
# noslip skipped
```

### 6.1 Warmstart (engine_forward.c:608–684, non-PGS branch)

```
qacc = qacc_warmstart                                  # from previous mj_advance (§7)
jar  = efc_J @ qacc_warmstart - efc_aref
(cost_ws, efc_force, efc_state, qfrc_constraint) = constraintUpdate(jar)   # §6.2
Ma = M @ qacc_warmstart                                # mju_mulSymVecSparse, §6.2a
cost_ws += Σ_i 0.5 * (Ma[i] - qfrc_smooth[i]) * (qacc_warmstart[i] - qacc_smooth[i])
(cost_smooth, …) = constraintUpdate(efc_b)             # Gauss(qacc_smooth) = 0
if cost_ws > cost_smooth: qacc = qacc_smooth
# (islands: unconstrained dofs forced to qacc_smooth — none here, nidof = nv)
```
The efc_force/qfrc_constraint values left over from these two calls are immediately
overwritten inside the solver; only `qacc` matters.

**Yes, mj_step+Euler uses `qacc_warmstart` saved at the end of the previous step's
`mj_advance` (engine_forward.c:936), which is the previous step's post-solver `d->qacc`
(NOT the implicitly-damped integration qacc).** First step after reset: warmstart = 0.

### 6.2 `mj_constraintUpdate_impl` (engine_core_constraint.c:2766–2959), this model

Inputs: `jar[nefc]` (= J·qacc − aref). ne = nf = 0, no elliptic rows ⇒ every row is the
simple "non-negative force" case:

```
cost = 0
for i in 0..nefc-1:
    if jar[i] >= 0:   force[i] = 0;            state[i] = SATISFIED
    else:             force[i] = -D[i]*jar[i]; state[i] = QUADRATIC
                      cost += 0.5 * D[i] * jar[i]²
qfrc_constraint = efc_Jᵀ @ force          # dense matTvec (in-solver version uses ctx arrays)
```

(6.2a) `mju_mulSymVecSparse` (engine_util_sparse.c:240–264) computes `M @ x` from the
lower CSR in one pass:
```
res = 0
for i in 0..8:
    res[i] += M[diag_i] * x[i]
    for each off-diag entry (i,j):  res[i] += M_ij * x[j];  res[j] += M_ij * x[i]
```

### 6.3 `mj_solPrimal` Newton main loop (engine_solver.c:1811–1968)

State per iteration: `qacc[9]`, `Ma[9] = M@qacc`, `Jaref[nefc] = J@qacc − aref`,
`efc_force/state`, `qfrc_constraint`, dense Cholesky factor `L[9][9]` (lower) of the
Hessian, `grad[9]`, `Mgrad[9]`, `search[9]`.

```
Ma    = M @ qacc                              # mulSymVecSparse
Jaref = J @ qacc - aref
PrimalUpdateConstraint:                       # solver.c:1010–1043
    constraintUpdate_impl(Jaref) → force, state, cost_constraint
    qfrc_constraint = Jᵀ force
    nactive = #(state != SATISFIED);  ncone = 0 (pyramidal)
    Gauss   = Σ_i 0.5*(Ma[i]-qfrc_smooth[i])*(qacc[i]-qacc_smooth[i])
    quadGauss[0] = Gauss;  cost = cost_constraint + Gauss
MakeHessian + FactorizeHessian(flg_recompute=0):     # §6.5
PrimalUpdateGradient:                          # solver.c:1047–1071
    grad  = Ma - qfrc_smooth - qfrc_constraint
    Mgrad = cholSolve(L, grad)                 # dense, §6.5
search = -Mgrad
scale  = 1 / Σ_i M[diag_i]                     # ISLAND scale = 1/trace(M)  (solver.c:1864–1871)
         # (monolithic path would use 1/(stat.meaninertia * nv) — NOT used here)

iter = 0
while iter < 100:
    alpha = PrimalSearch(tolerance = 1e-8 * 0.01, ls_iterations = 50)    # §6.4
    if alpha == 0: break
    qacc  += alpha * search
    Ma    += alpha * Mv                        # Mv computed inside PrimalSearch
    Jaref += alpha * Jv
    oldstate = state;  oldcost = cost
    PrimalUpdateConstraint (as above)
    HessianIncremental(oldstate)               # §6.5 rank-1 updates
    PrimalUpdateGradient (as above)
    improvement = scale * (oldcost - cost)
    gradient    = scale * ||grad||_2
    iter += 1
    if improvement < 1e-8 or gradient < 1e-8: break
    search = -Mgrad
qfrc_constraint, efc_force, qacc are the outputs;  solver_niter += iter
```

**Convergence scale (fp32-critical):** `scale = 1/trace(M)`; trace(M) ≈ Σ masses + leg
inertias ≈ O(30). `improvement` compares cost deltas of order (force·acc) against
1e-8 after scaling — in fp32 the subtraction `oldcost - cost` will hit cancellation
near convergence; compute costs in fp32 but consider keeping the cost accumulator and
this difference in higher precision (or accept earlier termination — it changes
trajectories vs double).

### 6.4 `PrimalSearch` — exact linesearch (engine_solver.c:1328–1511)

Precompute:
```
snorm = ||search||;  if snorm < 1e-15: return 0  (LSresult=1)
gtol     = (1e-8 * 0.01) * snorm / scale          # tolerance * ls_tolerance * snorm / scale
Mv = M @ search;  Jv = J @ search
PrimalPrepare (solver.c:1075–1147), pyramidal rows only:
    quadGauss[1] = dot(search, Ma) - dot(qfrc_smooth, search)
    quadGauss[2] = 0.5 * dot(search, Mv)
    per row i: quad[i] = ( 0.5*D[i]*Jaref[i]², D[i]*Jaref[i]*Jv[i], 0.5*D[i]*Jv[i]² )
```
Cost as function of α (`PrimalEval`, solver.c:1160–1294; ne=nf=0, no elliptic):
```
quadTotal = quadGauss
for each row i:  x = Jaref[i] + alpha*Jv[i];  if x < 0: quadTotal += quad[i]
cost      = quadTotal[2]*α² + quadTotal[1]*α + quadTotal[0]
deriv0    = 2*quadTotal[2]*α + quadTotal[1]
deriv1    = 2*quadTotal[2];   if deriv1 <= 0: deriv1 = 1e-15 (+warning)
each PrimalEval call increments LSiter
```
Search driver (a safeguarded Newton/bisection on the piecewise-quadratic 1-D function;
breakpoints are implicit — activity is re-tested per evaluation):
```
p0 = eval(0)
p1 = eval(p0.alpha - p0.deriv0/p0.deriv1)            # always one Newton step from 0
if |p1.deriv0| < gtol: return p1.alpha               # (alpha==0 → LSresult=2, returns 0)
dir = (p1.deriv0 < 0 ? +1 : -1)
# one-sided phase: repeat Newton steps from p1 while derivative keeps sign and magnitude > gtol
p2 = p0
while p1.deriv0*dir <= -gtol and LSiter < 50:
    p2 = p1
    p1 = eval(p1.alpha - p1.deriv0/p1.deriv1)
    if |p1.deriv0| < gtol: return p1.alpha
if LSiter >= 50: return p1.alpha                     # LSresult=3 (no bracket)
# bracketed phase: p1,p2 bracket the root of deriv0
p2next = p1
p1next = eval(p1.alpha - p1.deriv0/p1.deriv1)
while LSiter < 50:
    pmid = eval(0.5*(p1.alpha + p2.alpha))
    candidates = {p1next, p2next, pmid}
    if any candidate has |deriv0| < gtol: return the one with lowest cost
    b1 = updateBracket(p1, candidates, &p1next)      # solver.c:1298–1324:
    b2 = updateBracket(p2, candidates, &p2next)
    #   updateBracket(p, cands, pnext): for each cand, if p.deriv0 and cand.deriv0 share
    #   sign and |cand.deriv0| < |p.deriv0| → p = cand (flag). If updated:
    #   pnext = eval(p.alpha - p.deriv0/p.deriv1)
    if !b1 and !b2:
        return pmid.alpha                            # numerical floor (LSresult 0 or 7)
# out of iterations: return bracket end with best cost if it improves on p0, else 0
if p1.cost <= p2.cost and p1.cost < p0.cost: return p1.alpha    # LSresult=4
if p2.cost <= p1.cost and p2.cost < p0.cost: return p2.alpha    # LSresult=4
return 0                                                         # LSresult=5
```
`ls_tolerance` semantics: it scales `opt.tolerance` to form the absolute slope
threshold `gtol` (in cost-per-unit-alpha units, normalized by `snorm/scale`).
`ls_iterations` caps the total number of `PrimalEval` calls (LSiter), not loop trips.

### 6.5 Hessian assembly / factorization / updates (dense; engine_solver.c:1516–1807)

Initial (`MakeHessian` + `FactorizeHessian(flg_recompute=0)`):
```
Dvec[i] = (state[i] == QUADRATIC) ? efc_D[i] : 0
L(9x9)  = lower(Jᵀ diag(Dvec) J):       # mju_sqrMatTD_impl, engine_util_blas.c:886–921
    L = 0
    for j in 0..nefc-1:
        if Dvec[j] == 0: continue
        for i in 0..8:
            t = J[j][i];  if t == 0: continue
            L[i][0..i] += J[j][0..i] * (t * Dvec[j])        # lower triangle only
L += M (lower CSR scattered into dense lower)               # mju_addToSymSparse, sparse.c:217
mju_cholFactor(L, 9, mindiag=1e-15)     # engine_util_solve.c:33–62, in-place LLᵀ:
    for j in 0..8:
        t = L[j][j] - dot(L[j][0..j-1], L[j][0..j-1]);  if t < 1e-15: t = 1e-15 (rank--)
        L[j][j] = sqrt(t);  inv = 1/L[j][j]
        for i in j+1..8: L[i][j] = (L[i][j] - dot(L[i][0..j-1], L[j][0..j-1])) * inv
```
`cholSolve` (solve.c:66–91): forward substitution with division by `L[i][i]`, then
backward with `L[j][i]` (column access), standard.

Per-iteration (`HessianIncremental`, solver.c:1746–1807):
```
for i in 0..nefc-1:
    if oldstate[i] != QUAD and state[i] == QUAD:   up = +1
    elif oldstate[i] == QUAD and state[i] != QUAD: up = -1
    else: continue
    vec = J[i] * sqrt(efc_D[i])
    rank = mju_cholUpdate(L, vec, 9, up)     # solve.c:96–133, classic rank-1 (down)date:
        for k in 0..8 where vec[k] != 0:
            Lkk = L[k][k];  t = Lkk² ± vec[k]²;  if t < 1e-15 { t = 1e-15; rank-- }
            r = sqrt(t); c = r/Lkk; s = vec[k]/Lkk; L[k][k] = r
            for i>k: L[i][k] = (L[i][k] ± s*vec[i]) / c;  vec[i] = c*vec[i] - s*L[i][k]
            # NOTE the exact order: L-update uses OLD vec[i]; vec-update uses NEW L[i][k]
    if rank < 9: full recompute = MakeHessian-style rebuild + cholFactor; stop updating
```
**Bit-exactness:** the factor after rank-1 updates differs in rounding from a fresh
factorization. To replicate MuJoCo trajectories, implement `cholUpdate` exactly
(including the downdate clamp at 1e-15 and the rebuild-on-rank-loss path). fp32
warning: downdates (`up = −1`) lose accuracy fastest; the rebuild path is the safety
net — keep it.

---

## 7. `mj_Euler` (engine_forward.c:940–1016) — semi-implicit with implicit joint damping

EULERDAMP and DAMPER are enabled, and `dof_damping[3..8] > 0` ⇒ **the implicit branch
runs every step**:

```
mj_EulerSkip(skipfactor=0):
  # qH = M + h*diag(damping)   in CSR layout (same sparsity as M)
  qH[0..35] = M[0..35]
  for i in 0..8: qH[ M_rowadr[i] + M_rownnz[i] - 1 ] += timestep * dof_damping[i]
       # only dofs 3..8 actually change (root damping = 0)
  mj_factorI(qH, qHDiagInv, 9, M_rownnz, M_rowadr, M_colind, NULL)   # §2.4 factorization
  qfrc = qfrc_smooth + qfrc_constraint
  qacc_int = qfrc
  mj_solveLD(qacc_int, qH, qHDiagInv, 9, 1, ...)        # qacc_int = (M + hB)⁻¹ qfrc_total
  mj_advance(m, d, act_dot, qacc_int, qvel_arg=NULL)
```

So the exact discrete update is

> `v⁺ = v + h · (M + h·diag(damping))⁻¹ · (qfrc_smooth + qfrc_constraint)`

Implement that formula literally. It is *algebraically* the implicit-damping update:
since `qfrc_smooth` already contains the explicit damper force `−B·v` (via
`qfrc_damper`, §3.2), writing `f_total = f_nodamp − B·v` gives
`(M+hB)(v⁺−v) = h(f_nodamp − Bv)` ⇒ `(M+hB)v⁺ = M·v + h·f_nodamp`, the textbook
semi-implicit damped update. But port the computed form (solve for `qacc_int`, then
`v += h·qacc_int`), not the rearranged one — rounding differs.
Note `d->qacc` (the solver output) is left untouched; the damped
`qacc_int` is local to the integrator, while `d->qacc` is what gets saved to
`qacc_warmstart`.

`mj_advance` (engine_forward.c:830–937), reduced (no history, na=0, no sleep,
no plugins):

```
qvel += timestep * qacc_int                    # velocity first (semi-implicit)
# positions integrated with the NEW qvel (qvel_arg==NULL → d->qvel):
for j in 0..8: qpos[j] += timestep * qvel[j]   # mj_integratePosInd, support.c:642–677
                                               #   (slide & hinge are both plain scalar adds;
                                               #    loop is per body/joint but equals this)
time += timestep
qacc_warmstart = d->qacc                       # ← SOLVER qacc, not qacc_int
```

fp32 note: `qpos[0]` (rootx) grows unboundedly while increments stay ~1e-2·qvel;
fp32 absolute resolution at |x| = 100 is ~7.6e-6 — this is the single largest fp32
divergence driver for HalfCheetah. Consider fp64 (or compensated summation) for the
two root slide qpos accumulations only.

---

## 8. `mj_rnePostConstraint` (engine_core_smooth.c:2420–2624) — **OPTIONAL-SKIP**

In 3.6.0 this function is **lazily evaluated**: `mj_forwardSkip` only clears
`d->flg_rnepost`; the computation runs solely inside `mj_sensorAcc` / support readers
that need `cacc/cfrc_int/cfrc_ext` (engine_sensor.c:875–882, :1459, :1611–1613).
MODEL FACT: `nsensor = 0` ⇒ **it never executes in MuJoCo itself for this model**,
regardless of any envpool-era `post_constraint` setting. Nothing in obs/reward/state
(qpos, qvel, contacts, forces) depends on it. Divergence risk from skipping: **none**.

---

## 9. Reset path (`mj_resetData`, engine_io.c:1260–1430)

Relevant effects (everything else is diagnostics/zeroing of derived arrays):

```
qpos = qpos0 = (0,0,0,0,0,0,0,0,0)
qvel = 0;  ctrl = 0;  qacc = 0;  qacc_warmstart = 0
qfrc_applied = 0;  xfrc_applied = 0;  time = 0
ncon = ne = nf = nl = nefc = nisland = nidof = 0
```

Gym/envpool then writes `qpos = qpos0 + U(-.1,.1)⁹`, `qvel = N(0, .1)⁹` and calls
`mj_forward` (= §1 forward block without integration) so that derived quantities are
consistent before the first `mj_step`. The first `mj_step` therefore runs warmstart
with `qacc_warmstart = 0` (cost compare picks the better of 0-acc and qacc_smooth).
For determinism, replicate: reset → write state → full forward → loop of steps.

Note: at `qpos = qpos0` (torso z = .7, legs zeroed) **no contacts and no limit rows
are active** — first steps are free-fall with springs/dampers; nefc = 0 until landing
(bfoot/ffoot capsules reach the floor) or a limit is hit.

---

## 10. Summary of fp32-sensitive spots (consolidated)

1. `qpos[0]` unbounded growth (§7) — use fp64/Kahan for root slide positions.
2. Newton convergence tests `scale*(oldcost−cost) < 1e-8`, `scale*||grad|| < 1e-8`
   (§6.3): catastrophic cancellation in fp32; thresholds were tuned for double. With
   pure fp32, expect the solver to terminate by `alpha==0`/no-improvement instead —
   acceptable physically, not bit-comparable.
3. `gtol = 1e-10 * snorm/scale` (§6.4) is below fp32 resolution of `deriv0`; the
   linesearch will effectively run until `updateBracket` stalls (the
   `!b1 && !b2 → return pmid` exit). That exit is well-defined; keep it.
4. Cholesky downdates (§6.5) — keep the 1e-15→(fp32: use ~1e-12) clamp and the
   rank-loss rebuild.
5. `mju_makeFrame` tangent normalization near vertical capsule axes (§2.5.5).
6. Division floors: every `max(mjMINVAL, ·)` in §2.6.4 / §6 should use an fp32-scaled
   epsilon (1e-15 underflows to 0 risk-free in fp32 denormals, but `1/R` with
   R = 1e-15 overflows fp32: imp is clamped to ≤ 0.9999 and dA > 0 ⇒ in practice
   R ≥ ~1e-7·dA; still, clamp D = 1/R to fp32-finite).
7. `mj_factorI` has **no** diagonal clamp (§2.4) — assert positivity in debug.
8. Impedance `power()` with p=2 is exact multiply (§2.6.4) — no `pow` needed anywhere
   in the hot path.
9. `mju_isBad` thresholds (1e10) exceed fp32 max-safe range comfortably; fine as-is.

## 11. Static sizes for the kernel

```
nq = nv = 9, nbody = 8, njnt = 9, ngeom = 9, nu = 6, na = 0
ncon_max = 16 (8 pairs × 2), nl_max = 6 (one side per limited joint),
nefc_max = 6 + 16*4 = 70 rows; J dense 70×9; quad 70×3; Hessian/L dense 9×9 = 81;
M/qLD/qH CSR: 36 values + layout of §0.4.
```
