# AGENTS

## Alberta Plan Slide Law

### The 12 steps of the Alberta Plan

1. Representation I: Continual supervised learning with given features.
2. Representation II: Supervised feature finding.
3. Prediction I: Continual GVF prediction learning.
4. Control I: Continual actor-critic control.
5. Prediction II: Average-reward GVF learning.
6. Control II: Continuing control problems.
7. Planning I: Planning with average reward.
8. Prototype-AI I: One-step model-based RL with continual function approximation.
9. Planning II: Search control and exploration.
10. Prototype-AI II: The STOMP progression.
11. Prototype-AI III: Oak.
12. Prototype-IA: Intelligence amplification.

### The Alberta Plan's Research vision: Intelligence as signal processing over time

Four key features of DeepMind Alberta's AI research

1. Experience oriented. Experience is the data of AI
2. Online, temporally uniform, continually running & learning
3. Concern for computational cost, response time
4. Agents interacting with environments containing other agents

### Outline (take-home messages)

- Understanding intelligence is a grand scientific prize that may soon be within reach; the glory will go to the bold
- The Alberta Plan is a direct run at this great prize
- emphasizing intelligence as the acquisition of new knowledge
- My motivation for the plan stems from the Oak architecture, a complete conceptual design for a Model-based RL agent, a Proto-AI
- The Alberta Plan is a massive retreat; going back to the most basic algorithms; re-vamping them for continual and meta-learning (in 12 steps)

### First, the Big World Perspective

- The world is vastly more complex than the agent
- After all, the world contains many other agents, and what is going on in many of their minds matters for yours
- It follows that you can never accurately know the state of the world, much less its transition dynamics; your knowledge is a gross approximation of reality
- This alone makes prior knowledge a never-ending fool's errand, a mistake
- The agent must adapt its approximations to the particular part of the world that it encounters
- And do this continually as the part encountered changes
- The agent's approximations will be non-stationary; they must track; this is continual learning
- When learning is continual, extensive repetitive experience is obtained with learning; this makes meta-learning-learning how to learn-possible and essential
- Thus continual learning and meta-learning (e.g., representation learning) are naturally linked

### The Alberta Plan's model of the intelligent agent

- Adds SubTasks (≈feature-control subtasks)
- Multiple policies and value functions

### The common model of the intelligent agent

The agent comprises four components:

- Perception produces the state representation used by all components
- Reactive Policy quickly produces an action appropriate to the state
- Value Function evaluates how well things are going, and changes the policy (learning)
- Transition model predicts the consequences of alternate actions, and changes the policy (planning)

### The Alberta Plan eschews domain knowledge

- The Plan should contain nothing specific to the human world
- nothing about vision, space, objects, language, or other agents
- nothing that distinguishes people from other animals
- nothing about specific ecological niches
- No content, just algorithms for acquiring and organizing knowledge

Learning first, domain knowledge last.
Why?

- To keep the problem small, the methods scalable
- To keep the focus on acquiring new knowledge

### The Alberta Plan takes the conventional RL interface seriously

- A complex computational agent interacts with a vastly more complex environment to maximize its reward
- optimality is impossible
- approximation is required
- tracking is required
- The interaction is continual and temporally uniform
- there are no special training periods
- there are no episodes, no discounting
- Actions and observations are just signals with no intrinsic meaning; they are just information
- no language, vision, space, objects, or geometry

### The Oak Architecture (Options And Knowledge) is based on a cycle of discovery

- Models of options
- State features
- Subtasks to attain features
- Options/values that solve (sub)tasks
- Arrows show the direction of primary flow
- Each arrow has a slower backward flow of credit
- "I use you, so don't change too much"
- The consumers of state and time abstractions support, evaluate, and shape the abstractions
- Cycle of discovery produces abstractions tailored to the environment and ultimately tied to reward

## Hard Rules

- Project is greenfield. It should never have legacy code or legacy fallbacks. Replace or refactor old paths instead of adding compatibility fallbacks.

## Paper-Grounded Corollaries

- Benchmark-driven patching is forbidden.
- OaK must stay online, temporally uniform, and average-reward.
- No train/eval split, no discounting, and no episodic special cases beyond target gating at termination.
- The agent must keep one shared state representation used by perception, reactive policy, value functions, transition models, subtasks, options, and planning.
- Oak must include reward-respecting subtasks, multiple policies and value functions, and models of options used by planning.
- Domain knowledge is forbidden. No environment-specific heuristics, no benchmark-specific shaping, and no task-identity-conditioned control logic.
- Computational cost and response time are first-class. Use external profiling and the full benchmark truth surface to localize bottlenecks; do not grow repo-resident diagnostic frameworks in the hot path.
