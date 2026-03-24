# ClueBot RL TODO List

## Notes & Insights

- RL agent is backed by EliminationBot logic — belief matrix from direct observation only,
  no heuristic strategy baked in. This gives RL room to discover strategy rather than
  fighting against an already-optimal heuristic.
- HeuristicsBot is a separate research track: an exploration of what we believe is the
  optimal strategy (entropy-based suggestion + Bayesian inference). Not used as RL backing.
- CFR is a third benchmark track to test whether HeuristicsBot's strategy is actually optimal
  or just approximately optimal.
- RL agent has three heads: suggestion, reveal, accusation timing. Each learns something
  distinct that the heuristic cannot easily capture.
- Suggestion head gets guided exploration (weighted toward high solution-probability cards)
  rather than pure epsilon-random, so it stays in productive regions of action space.
- Accusation timing head takes over from the hardcoded threshold policy. Proxies opponent
  progress from public log (unanswered suggestions, turn count, opponents still in game).
- Reveal head stays as is — random heuristic has no good answer here, RL adds real value.
- State representation: hand, EliminationBot belief matrix, compact suggestion history,
  opponent progress proxies for accusation head.
- Reward shaping and accusation logic remain configurable in ClueEnv.
- Do not model opponent belief matrices directly — too expensive and private. Use public
  log proxies instead.
- Use Docker and AWS for deployment (not Kubernetes unless scaling is needed).

## Architecture TODOs

### RL Agent

- [x] Expose Clue game logic as a Gym-like RL environment (ClueEnv)
- [x] Add get_state() with hand, belief matrix, and suggestion history
- [x] Expose suggestion history for backward inference
- [x] Make accusation logic and reward shaping configurable in ClueEnv
- [x] Implement dual-head DQN (suggestion + reveal) using PyTorch
- [x] Implement training loop with experience replay
- [x] Implement reward shaping (information gain, terminal cleanliness)
- [ ] Swap RL agent backing from HeuristicsBot to EliminationBot
- [ ] Add guided exploration to suggestion head (weight epsilon-random toward
      high solution-probability actions instead of pure uniform random)
- [ ] Implement accusation timing head (third head on shared trunk)
      - Input: full game state + opponent progress proxies
      - Output: binary accuse/wait
      - Proxies: turn count, opponents still in, unanswered suggestion counts per opponent
- [ ] Compact state encoding for suggestion history (replace one-hot with
      dense 12-float per entry: solution probs of suggested cards + normalized player flags)
- [ ] Evaluate RL agent against EliminationBot, HeuristicsBot, CFR baselines

### HeuristicsBot (Strategy Research Track)

- [x] Bayesian belief matrix + entropy-based suggestion selection
- [x] Solution card tracking (checkForSolutionCards, cascading deductions)
- [x] Backward inference (deduce shown card when only one possibility remains)
- [ ] Fix double-update bug in ClueEnv.step() — crossOff + processNewSuggestions
      both processing the same suggestion record
- [ ] Audit normalizeCardAcrossPlayers calls to ensure Solution is not excluded
      incorrectly from normalization
- [ ] BluffBot — suggests own cards to suppress information leakage, actively
      reinforces opponents' mistaken beliefs
- [ ] MirrorBot — mirrors recent opponent suggestions to obscure its own knowledge state
- [ ] PolicySwitchingBot — switches between strategies based on inferred game phase

### CFR Benchmark Track

- [ ] Model Clue as an extensive-form game with imperfect information
- [ ] Implement vanilla CFR over the game tree
- [ ] Compare CFR suggestion selection vs HeuristicsBot entropy approach —
      if CFR wins, HeuristicsBot is not optimal; if they match, entropy is near-optimal
- [ ] Track opponent tendencies (bluffing frequency, suggestion reuse) and feed
      into CFR strategy

## Deployment TODOs

- [ ] Build API/backend for deployment (Flask/FastAPI)
- [ ] Build web or CLI frontend for users
- [ ] Containerize app with Docker
- [ ] Deploy to AWS
- [ ] (Optional) Add user authentication, game history, leaderboards

---
Add new ideas, notes, or tasks as the project evolves!
