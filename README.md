# ClueBot: Reinforcement Learning Clue Agent

## Project Overview

ClueBot is an AI agent designed to play the board game Clue using reinforcement
learning and probabilistic reasoning. The project uses PyTorch for deep Q-learning,
a modular belief tracking system, and a game engine that supports multiple bot
strategies for training, evaluation, and benchmarking.

The project has three parallel tracks:

**RL Agent** — A three-head DQN backed by EliminationBot belief tracking. Learns
suggestion strategy, reveal selection, and accusation timing from game outcomes
rather than inheriting strategy from a heuristic.

**HeuristicsBot** — A research track into what we believe is the optimal Clue
strategy: entropy-based suggestion selection on top of a Bayesian belief matrix
with cascading deductions. Serves as a benchmark and a hypothesis about optimal play.

**CFR** — A game-theoretic benchmark to test whether HeuristicsBot's strategy is
actually optimal or just approximately optimal. If CFR's suggestion selection
matches HeuristicsBot's, entropy-based play is near-optimal. If it diverges,
that divergence reveals something non-obvious.

---

## Architecture

```text
ClueBasics/
  GameRules.py        — game engine (dealing, suggestions, accusations, logging)
  Player.py           — abstract base class for all players
  Card.py             — card representation

agents/
  EliminationBot.py   — direct observation belief tracking, random suggestion
  HeuristicsBot.py    — Bayesian belief matrix + entropy-based suggestion (strategy research)
  TriggerHappyBot.py  — accuses whenever no one responds to a suggestion
  BluffBot.py         — (planned) suggests own cards, reinforces mistaken beliefs
  MirrorBot.py        — (planned) mirrors opponent suggestions to obscure knowledge state
  PolicySwitchingBot.py — (planned) switches strategy based on inferred game phase

rl/
  clue_env.py         — Gym-style environment wrapping GameRules
  dqn_agent.py        — three-head DQN (suggestion + reveal + accusation timing)
  train_dqn.py        — training loop
  reward_schedules.py — reward shaping functions
```

---

## Key Design Decisions

### 1. RL Agent Backing: EliminationBot, Not HeuristicsBot

The RL agent's belief matrix is provided by **EliminationBot**, not HeuristicsBot.
This is a deliberate choice.

HeuristicsBot already implements what we believe is the optimal suggestion strategy
(entropy-based selection). Using it as the RL backing means the DQN is fighting
against a strategy that is already near-optimal — there is no room to learn anything
meaningful about suggestion selection, and as epsilon decays, the DQN overrides
good heuristic decisions with worse learned ones.

EliminationBot provides belief tracking grounded in direct observation only, with
no strategy embedded. This gives the suggestion head genuine room to discover why
high-entropy, high-probability suggestions are good, rather than inheriting that
knowledge and then corrupting it.

The suggestion head may converge to something similar to HeuristicsBot's strategy.
That is a valid and interesting result — it means the RL agent rediscovered the
optimal heuristic from game outcomes alone.

---

### 2. Three-Head DQN Architecture

The DQN network has a shared trunk with three output heads:

**Suggestion Head**
Q-values over all (suspect, weapon, room) combinations. Primary game decision.
Uses guided exploration: during random action selection, weights toward high
solution-probability actions rather than pure uniform random. This keeps exploration
in productive regions of the action space without hardcoding strategy.

**Reveal Head**
Q-values over MAX_HAND_SIZE (7) card slots. Decides which matching card to show
when the agent holds multiple cards that match an opponent's suggestion. Trunk
gradients are detached during reveal-only updates so the reveal head cannot corrupt
shared features. Reveal transitions accumulate during an episode and are stored
retrospectively once the terminal reward is known.

**Accusation Timing Head** *(novel contribution)*
Binary output: accuse now vs wait one more turn. Replaces the hardcoded certainty-
threshold policy. The head takes the full game state plus opponent progress proxies:
turn count, opponents still in game, per-opponent unanswered suggestion counts
(a proxy for how close each opponent is to solving). Does not require opponent
belief matrices — those are private and intractable to compute. Public log proxies
are sufficient for learning when to gamble on an early accusation.

This framing — accusation as a learned policy under uncertainty rather than a
deterministic threshold — is the primary novel contribution of the RL track.

---

### 3. State Representation

```text
[hand_encoding | elimination_belief_matrix | compact_suggestion_history | opponent_proxies]
```

**hand_encoding** (`action_space_size`): binary, 1 if agent holds any card in that
action tuple.

**elimination_belief_matrix** (`num_owners * num_cards`): from EliminationBot,
grounded in direct observation only. No Bayesian inference or heuristic updates.

**compact_suggestion_history** (last N=5 suggestions, 12 floats each):
- Solution probabilities of the three suggested cards
- Normalized suggester index
- Was refuted flag
- Suggester is RL flag
- Skipped player count (normalized)
- Normalized responder index

This replaces the old one-hot history encoding (1620 sparse dimensions for 5 entries)
with 60 dense floats, dramatically reducing state dimensionality.

**opponent_proxies** (for accusation timing head):

- Turn count (normalized)
- Number of opponents still in game (normalized)
- Per-opponent unanswered suggestion count (proxy for solution convergence)

---

### 4. Accusation Timing

The RL agent does not have "accuse" in its suggestion action space. Accusations
are triggered by the accusation timing head, which outputs a binary accuse/wait
decision each turn after the suggestion is processed.

The head learns to balance confidence in the current belief state against the risk
of letting an opponent win. Early accusation at 80% confidence may be correct if
an opponent has made several unanswered suggestions — the heuristic threshold of
99% ignores this context entirely.

---

### 5. Reveal Decision Hook

When an opponent suggests and the RL agent holds matching cards, the DQN reveal
head picks which card to show. The hook flows as:

```text
ClueEnv.step()
  → _make_reveal_chooser()      captures current state in a closure
  → _run_bot_turn(bot, chooser) monkey-patches game.makeSuggestion temporarily
  → GameRules.makeSuggestion()  passes chooser to responding player
  → Player.refuteSuggestion()   passes chooser to showCard()
  → Player.showCard()           calls chooser(matching_cards) if provided
  → DQNAgent.select_reveal()    queries reveal head, returns chosen card
```

Non-RL players always use their own `_choose_card_to_show()` logic (default: random).

---

### 6. HeuristicsBot: Strategy Research Track

HeuristicsBot is not a training opponent — it is a separate investigation into
optimal Clue play. Its design hypothesis: suggesting the card with highest solution
probability, tiebroken by highest entropy, is the optimal information-gathering
strategy.

This hypothesis will be tested against CFR. If CFR produces similar suggestion
distributions, entropy-based play is near-optimal. If CFR diverges, the divergence
reveals non-obvious structure in Clue strategy worth studying.

Future extensions: BluffBot and MirrorBot as adversarial training opponents that
test whether optimal play is robust to deceptive opponents.

---

### 7. Reward Signals

| Signal              | Value  | When                         |
|---------------------|--------|------------------------------|
| Win                 | +1.0   | Correct accusation           |
| Wrong accusation    | -1.0   | Incorrect accusation         |
| Bot wins            | -1.0   | Any opponent wins first      |
| Info gain (shaped)  | ±small | Non-terminal steps only      |

Terminal rewards are never modified by shaping. This prevents the agent from
treating a wrong accusation as merely "information gathering with a small penalty."

---

### 8. Training Mechanics

**Double DQN**: policy network selects next action, target network evaluates it.
Reduces Q-value overestimation in sparse-reward environments.

**Huber loss** (`smooth_l1_loss`): more stable than MSE when Q-value errors are
large early in training.

**Per-episode epsilon decay**: decays once per episode, not per step. Per-step
decay collapsed exploration within ~15 episodes.

**Guided exploration**: suggestion head epsilon-random samples are weighted toward
high solution-probability actions, not pure uniform. Keeps random play informative.

**Retrospective reveal storage**: reveal transitions accumulate during an episode.
Terminal reward is assigned to all reveal transitions after the episode ends, once
win/loss is known.

---

## Bot Hierarchy

| Bot                | Belief Tracking     | Suggestion Strategy        | Role                         |
|--------------------|---------------------|----------------------------|------------------------------|
| TriggerHappyBot    | None                | Random, accuse if no refute| Baseline / training fodder   |
| EliminationBot     | Direct observation  | Random                     | RL backing / weak baseline   |
| HeuristicsBot      | Bayesian + cascade  | Entropy-maximizing         | Strategy research benchmark  |
| CFR Agent          | Full game tree      | Regret-minimizing          | Optimality benchmark         |
| RL Agent           | EliminationBot      | Learned (3-head DQN)       | Primary research subject     |
| BluffBot           | Bayesian            | Deceptive                  | Adversarial training opponent|
| MirrorBot          | Bayesian            | Mimicry-based              | Adversarial training opponent|

---

## Reward Signals

| Signal                | Value  | When                    |
| --------------------- | ------ | ----------------------- |
| Win                   | +1.0   | Correct accusation      |
| Wrong accusation      | -1.0   | Incorrect accusation    |
| Bot wins              | -1.0   | Any opponent wins       |
| Info gain (shaped)    | ±small | Every non-terminal step |

---

## Getting Started

```bash
git clone <repo-url>
cd ClueBot
pip install -r requirements.txt
python train_dqn.py
```

To resume from a checkpoint:

```bash
# Automatically picks up checkpoints/latest.pth if it exists
python train_dqn.py
```

---

## Roadmap Status

- [x] Phase 1: Core game simulation
- [x] Phase 2: Information tracking & belief modeling
- [x] Phase 3 (partial): Bot architecture (TriggerHappy, Elimination, Heuristics)
- [x] Phase 4 (partial): RL integration (dual-head DQN, env wrapper, training loop)
- [ ] Phase 3 remaining: BluffBot, MirrorBot, PolicySwitchingBot
- [ ] Phase 4 remaining: EliminationBot backing, accusation timing head,
      guided exploration, compact state encoding
- [ ] Phase 5: CFR benchmark, opponent tendency modeling
- [ ] Phase 6: Flask/FastAPI backend, web frontend, Docker, AWS
