# ClueBot Roadmap

---

## Phase 1: Core Game Simulation ✅

**Goal:** Build a functional virtual Clue that bots can play.

- Implement GameRules class
- Define Player abstract base class
- Randomly select solution cards, shuffle and deal remainder
- Turn-based game loop: suggestion → response → accusation
- Track card visibility per player
- Suggestion log: who suggested what, who responded, what was shown
- Correct accusation → win condition

---

## Phase 2: Information Tracking & Belief Modeling ✅

**Goal:** Give bots memory and reasoning capabilities.

- Per-player suggestion history (public log with visibility rules)
- Card likelihood matrix: P(owner has card) for every (owner, card) pair
- Backward inference: deduce shown card when only one possibility remains
- Cascading deductions: when a card is confirmed in solution, eliminate others in category

---

## Phase 3: Bot Architecture 🔄

**Goal:** Multiple bot strategies ranging from naive to optimal, plus adversarial bots.

### Completed

- TriggerHappyBot — accuses whenever no one responds to a suggestion
- EliminationBot — crosses off cards through direct observation only
- HeuristicsBot — Bayesian belief matrix + entropy-based suggestion + solution card tracking

### HeuristicsBot: Strategy Research Track

HeuristicsBot is not just a training opponent — it is an active research track into
what the optimal Clue strategy looks like. The hypothesis is that entropy-based
suggestion selection (highest solution probability, tiebroken by highest entropy)
is near-optimal. CFR will be used to test this.

Remaining:

- [ ] BluffBot — suggests own cards to mislead opponents, actively reinforces mistaken beliefs
- [ ] MirrorBot — mirrors recent opponent suggestions to obscure its own knowledge state
- [ ] PolicySwitchingBot — switches strategies based on inferred game phase

---

## Phase 4: Reinforcement Learning Integration 🔄

**Goal:** Train an RL agent that discovers strategy from game outcomes rather than
inheriting it from a heuristic.

### Architecture

The RL agent is backed by **EliminationBot** logic for belief tracking. This is a
deliberate choice: EliminationBot's belief matrix is grounded in direct observation
only, with no strategy baked in. This gives the RL agent room to discover suggestion
strategy rather than fighting against an already-optimal heuristic.

**HeuristicsBot is not used as RL backing.** It is a separate benchmark.

The DQN network has a **shared trunk** with **three output heads**:

#### Suggestion Head**

- Q-values over all (suspect, weapon, room) combinations
- Guided exploration: epsilon-random weighted toward high solution-probability
  actions, not pure uniform random. Keeps exploration in productive regions.
- The head may converge to something like HeuristicsBot's strategy — that is a
  valid and interesting result, not a failure.

#### Reveal Head**

- Q-values over MAX_HAND_SIZE card slots
- Decides which matching card to show when forced to respond to an opponent suggestion
- Trained with detached trunk to avoid interfering with suggestion head features
- Retrospective reward assignment: reveal transitions accumulate during episode,
  terminal reward assigned after game ends

**Accusation Timing Head** *(new)*

- Binary output: accuse now vs wait
- Replaces the hardcoded certainty-threshold policy
- Input includes opponent progress proxies from the public log:
  - Turn count
  - Number of opponents still in game
  - Per-opponent unanswered suggestion count (proxy for how close they are)
  - Your own solution probabilities and possibleSuspects/Weapons/Rooms lengths
- Does not require opponent belief matrices — those are private and intractable.
  Public log proxies are sufficient.
- This is the novel contribution: accusation as a learned policy under uncertainty
  rather than a deterministic threshold.

### State Representation

```text
[hand_encoding | elimination_belief_matrix | compact_suggestion_history | opponent_proxies]
```

- hand_encoding: binary over action space
- elimination_belief_matrix: from EliminationBot (direct observation only)
- compact_suggestion_history: last N suggestions as 12 dense floats each
  (solution probs of suggested cards + normalized player/responder flags)
  rather than one-hot over action space (massive dimensionality reduction)
- opponent_proxies: turn count, opponents remaining, unanswered suggestion counts

### Reward Signals

| Signal              | Value  | When                         |
|---------------------|--------|------------------------------|
| Win                 | +1.0   | Correct accusation           |
| Wrong accusation    | -1.0   | Incorrect accusation         |
| Bot wins            | -1.0   | Any opponent wins first      |
| Info gain (shaped)  | ±small | Non-terminal steps only      |

Terminal rewards are never modified by shaping.

### CompletedCode

- [x] ClueEnv Gym-style wrapper
- [x] get_state(), get_legal_actions(), step()
- [x] Dual-head DQN (suggestion + reveal)
- [x] Training loop with experience replay, Double DQN, Huber loss
- [x] Per-episode epsilon decay
- [x] Reward shaping (information gain, terminal cleanliness)
- [x] Retrospective reveal reward assignment

### Remaining

- [ ] Swap RL backing from HeuristicsBot to EliminationBot
- [ ] Guided exploration for suggestion head
- [ ] Accusation timing head implementation
- [ ] Compact suggestion history encoding
- [ ] Opponent progress proxies in state
- [ ] Evaluate against EliminationBot, HeuristicsBot, CFR baselines

---

## Phase 5: Game Theory & Advanced Inference

**Goal:** CFR as a benchmark and opponent modeling.

### CFR Benchmark

Model Clue as an extensive-form game with imperfect information and implement
vanilla CFR. Primary purpose: determine whether HeuristicsBot's entropy-based
strategy is actually optimal or just approximately optimal. If CFR's suggestion
selection matches HeuristicsBot's, the entropy approach is likely near-optimal.
If CFR diverges, that divergence reveals something non-obvious about Clue strategy.

### Opponent Modeling

- Track per-opponent tendencies from the public log (bluffing frequency,
  suggestion reuse patterns, aggression toward early accusation)
- Feed tendency features into accusation timing head as additional input
- Long-term: use opponent models to inform suggestion selection (suggest cards
  that exploit a known bluffing opponent's information leakage)

---

## Phase 6: Deployment

- Flask/FastAPI backend
- Web frontend (play against bots)
- Docker containerization
- AWS deployment
- (Optional) User authentication, game history, leaderboards

---

## Bot Hierarchy Summary

| Bot                | Belief Tracking     | Suggestion Strategy        | Role                        |
|--------------------|--------------------|-----------------------------|------------------------------|
| TriggerHappyBot    | None               | Random, accuse if no refute | Baseline / training fodder   |
| EliminationBot     | Direct observation | Random                      | RL backing / weak baseline   |
| HeuristicsBot      | Bayesian + cascade | Entropy-maximizing          | Strategy research benchmark  |
| CFR Agent          | Full game tree     | Regret-minimizing           | Optimality benchmark         |
| RL Agent           | EliminationBot     | Learned (3-head DQN)        | Primary research subject     |
| BluffBot           | Bayesian           | Deceptive                   | Adversarial training opponent|
| MirrorBot          | Bayesian           | Mimicry-based               | Adversarial training opponent|
