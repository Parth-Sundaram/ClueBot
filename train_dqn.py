import os
import numpy as np
import torch
import logging
from clue_env import ClueEnv
from dqn_agent import DQNAgent
from reward_schedules import info_gain_reward_shaping

# ------------------------------------------------------------------
# Debug mode
VERBOSE = False  # Set to True to see full game output for debugging single episodes

# ------------------------------------------------------------------
# Logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# ------------------------------------------------------------------
# Hyperparameters
NUM_EPISODES        = 3000
MAX_STEPS           = 200
UPDATE_TARGET_EVERY = 20        # episodes
EVAL_EVERY          = 100       # episodes
EVAL_EPISODES       = 30
CHECKPOINT_EVERY    = 100       # episodes
CHECKPOINT_DIR      = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Environment & agent

env = ClueEnv(reward_shaping=info_gain_reward_shaping, verbose=VERBOSE)
state_dim      = env.compute_state_dim()
suggestion_dim = len(env.action_space)

print(f"State dim: {state_dim} | Suggestion dim: {suggestion_dim}")

agent = DQNAgent(
    state_dim=state_dim,
    suggestion_dim=suggestion_dim,
    lr=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.997,
    buffer_size=20000,
    reveal_buffer_size=5000,
    batch_size=64,
)

# Wire agent into env so reveal head can be called during opponent turns
env.set_agent(agent)

# # --- DEBUG ---
# state = env.reset()
# rl = env.players[env.rl_player_index]
# print("Hand:", [c.name for c in rl.cards])
# print("Possible suspects:", [c.name for c in rl.possibleSuspects])
# print("Possible weapons:", [c.name for c in rl.possibleWeapons])
# print("Possible rooms:", [c.name for c in rl.possibleRooms])
# print("P(Solution, Professor Plum):",
#       rl.getProbability("Solution", rl.game.suspectCards["Professor Plum"]))
# print("P(Solution, Kitchen):",
#       rl.getProbability("Solution", rl.game.roomCards["Kitchen"]))
# # --- END DEBUG ---

# Resume from checkpoint if available
latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
start_episode = 0
if os.path.exists(latest_ckpt):
    agent.load(latest_ckpt)
    meta_path = latest_ckpt + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            start_episode = int(f.read().strip())
    print(f"Resumed from checkpoint at episode {start_episode}, epsilon={agent.epsilon:.3f}")

# ------------------------------------------------------------------
# Evaluation

def evaluate(agent, env, episodes):
    eps_backup  = agent.epsilon
    agent.epsilon = 0.0
    rewards, wins = [], 0
    for _ in range(episodes):
        state       = env.reset()
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            legal  = env.get_legal_actions()
            action = agent.select_suggestion(state, legal)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                if info.get("result") == "win":
                    wins += 1
                break
        rewards.append(total_reward)
    agent.epsilon = eps_backup
    return np.mean(rewards), wins / episodes

# ------------------------------------------------------------------
# Training loop

all_rewards  = []
episode_wins = []
eval_scores  = []

for episode in range(start_episode, NUM_EPISODES):
    state        = env.reset()
    total_reward = 0.0
    won          = False
    episode_reveal_transitions = []  # accumulate reveal transitions this episode

    for step in range(MAX_STEPS):
        legal         = env.get_legal_actions()
        action        = agent.select_suggestion(state, legal)
        next_state, reward, done, info = env.step(action)

        # --- Suggestion transition ---
        agent.store_suggestion(state, action, reward, next_state, done)
        agent.update_suggestion()

        # --- Reveal transition (if the RL agent revealed a card this step) ---
        for reveal in (env._last_reveal or []):
            episode_reveal_transitions.append({
                **reveal,             # ← the individual dict
                "next_state": next_state,
                "done":       done,
                "reward_placeholder": reward,
            })

        state         = next_state
        total_reward += reward

        # Update reveal head each step too (if buffer has enough)
        agent.update_reveal()

        if done:
            won = info.get("result") == "win"
            break

    # Per-episode summary
    if VERBOSE or (episode + 1) % 10 == 0:
        result = info.get("result", "max_steps")
        print(f"Ep {episode+1}: {result} in {step+1} steps | reward {total_reward:+.3f}")

    # --- Retrospective reveal storage ---
    # Now we know the terminal reward for this episode. Assign it to all
    # reveal transitions that occurred this episode so the reveal head
    # learns from the actual game outcome.
    terminal_reward = 1.0 if won else (-1.0 if total_reward < 0 else 0.0)
    for t in episode_reveal_transitions:
        agent.store_reveal(
            t["state"],
            t["card"],
            t["all_cards"],
            terminal_reward,    # win/loss signal propagated back
            t["next_state"],
            t["done"],
        )

    agent.decay_epsilon()

    all_rewards.append(total_reward)
    episode_wins.append(int(won))

    if (episode + 1) % UPDATE_TARGET_EVERY == 0:
        agent.update_target()

    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(all_rewards[-50:])
        win_rate   = np.mean(episode_wins[-50:])
        msg = (f"Episode {episode+1:>5} | "
               f"Avg Reward: {avg_reward:+.3f} | "
               f"Win Rate: {win_rate:.1%} | "
               f"Epsilon: {agent.epsilon:.3f} | "
               f"Reveal buffer: {len(agent.reveal_memory)}")
        logging.info(msg)
        print(msg)

    if (episode + 1) % EVAL_EVERY == 0:
        avg_eval, eval_win_rate = evaluate(agent, env, EVAL_EPISODES)
        eval_scores.append((episode + 1, avg_eval, eval_win_rate))
        eval_msg = (f"[EVAL] Episode {episode+1:>5} | "
                    f"Avg Score: {avg_eval:+.3f} | "
                    f"Win Rate: {eval_win_rate:.1%}")
        logging.info(eval_msg)
        print(eval_msg)

    if (episode + 1) % CHECKPOINT_EVERY == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{episode+1}.pth")
        agent.save(ckpt_path)
        agent.save(latest_ckpt)
        with open(latest_ckpt + ".meta", "w") as f:
            f.write(str(episode + 1))
        logging.info(f"Checkpoint saved: {ckpt_path}")

# ------------------------------------------------------------------
final_path = "dqn_cluebot_final.pth"
agent.save(final_path)
print(f"\nTraining complete. Model saved to {final_path}")
