# OaK — Options And Knowledge (ROUGH DRAFT)

## An implementation of [the Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) (Sutton, Bowling & Pilarski, 2022)


## Eyes on the Prize (Sutton, [incompleteideas.net](http://incompleteideas.net/))

The Alberta Plan exists in service of a single long-term goal Sutton calls
*the Prize*: understanding the principles of intelligence well enough to
create (or become) beings of greater intelligence. In that framing, intelligence is
*"the computational part of an agent's ability to predict and control a
stream of sensations, particularly a designated numerical sensation (called
reward), while interacting with a vastly more complex world."* Everything
the agent ever knows arrives through the **narrow aperture of experience** —
the alternating stream of actions, sensations, and reward — and it must do
its work in real time, against a world it can never fully contain.

OaK is a working bet on the four commitments that follow from that goal:

1. **Intelligence as real-time signal processing to maximize reward** under
   bounded compute — not offline optimization on a curated dataset.
2. **Continual learning from ordinary, unprepared experience** — no
   training/eval split, no episode reset inside the agent, no demonstrations
   or special training distributions.
3. **[The Big-World Hypothesis](https://openreview.net/pdf?id=Sv7DazuCn8)** —
   the agent is and always will be vastly less complex than its world, so
   every component must approximate, track, and be revisable, even in the
   limit.
4. **Continual construction of abstractions** — features, subtasks, options,
   GVFs — so the agent can think simply about long spans of time without
   collapsing them into a fixed scale.



### Sutton and Bowling's architecture addresses "The Problem With AI Nobody Wants to Face"

[![Watch on YouTube](https://img.youtube.com/vi/qqXSpWYSxBw/hqdefault.jpg)](https://www.youtube.com/shorts/qqXSpWYSxBw)


## The FC-STOMP cycle / OaK Architecture

> *"Understanding intelligence is a grand scientific prize that may soon be
> within reach; the glory will go to the bold."*
> — Sutton, Alberta Plan slides, p. 12

The architecture for the Step-11 agent is the FC-STOMP cycle: the paper names
the cycle STOMP and the architecture Oak; FC (Feature Construction) is
prepended here because under the Big-World commitment features themselves must
be continually constructed (Plan Steps 1–2) rather than given. The full live
loop is exposed verbatim as named stages in
[`Sources/OakKernel.metal`](Sources/OakKernel.metal):

| Stage | Role |
|---|---|
| **F**eature **C**onstruction | Continual feature finding + GVFPrediction over feature cumulants; features are generated, tested, and replaced by utility (Plan Steps 1–3, 5; Step-11 feedback) |
| **S**ub**T**ask | Reward-Respecting subtasks (stopping-value per [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4); low-utility subtasks are removed and replaced (Step-11 feedback) |
| **O**ption | Option policies + RR-derived continuation; the **option keyboard** (chord λ ∈ R^K) blends component options into the emitted action (Step-11 addition) |
| **M**odel | Option models r̂(s,o), n̂(s,o), and average-reward duration l̂(s,o); models that never pay off in planning are deleted with their option (Step-11 feedback) |
| **P**lanning | Average-reward option backup `Q = rHat − ρ·lHat + V(nHat)`, driven by search-control prioritization |

## Install / build
Please ask your agent to set this up. You must have an Apple silicon machine

## Wait? No PyTorch? Lets Talk about that.

This repo is mostly maintained by agents, so we need to go **fast**. Why rely on an unoptimized framework when we can craft accelerated GPU kernels with Claude directly? We use Xcode traces and compare OaK's outputs bit for bit, so we can losslessly upgrade and optimize the kernel without large dependencies or frameworks getting in the way. The code is the algorithmic logic for OaK plus a thin launch script, and that's it.


## John Carmack: "AGI is likely tens of thousands of lines of code, not millions"

Carmack's ideas line up with Sutton's, and he is even working with Amii.

https://www.amii.ca/updates-insights/john-carmack-and-rich-sutton-agi

I follow John Carmack's philosophy of AGI: "I am not a madman for saying that Artificial General Intelligence is likely to be tens of thousands of lines of code, not millions."

[![Watch on YouTube](https://img.youtube.com/vi/6gi8TGNpS_E/hqdefault.jpg)](https://www.youtube.com/watch?v=6gi8TGNpS_E&t=73s)

▶ Watch on YouTube (jumps to 1:13): <https://www.youtube.com/watch?v=6gi8TGNpS_E&t=73s>
