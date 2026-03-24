import numpy as np


def info_gain_reward_shaping(prev_state, state, action, base_reward, env):
    """
    Shape the non-terminal reward by the reduction in belief matrix entropy
    after a suggestion.

    Terminal rewards (+1 win, -1 loss) are never modified — the win/loss
    signal must remain clean for both the suggestion and reveal heads.

    State vector layout (must match clue_env.get_state):
      [hand (action_size) | belief_matrix (owners * cards) | suggestion_history (...)]
    """
    # Never touch terminal rewards
    if base_reward != 0:
        return base_reward

    # Also skip if accusation policy would fire this step — the -1/-1
    # penalty must land clean, not get washed out by entropy gain.
    rl_player = env.players[env.rl_player_index]
    if env._default_accusation_policy(rl_player):
        return base_reward

    action_size = len(env.action_space)
    owners      = env.players[env.rl_player_index].owners
    belief_len  = len(owners) * len(env.game.cards)

    prev_belief = prev_state[action_size: action_size + belief_len]
    curr_belief = state[action_size:      action_size + belief_len]

    def entropy(probs):
        probs = np.clip(probs, 1e-8, 1.0)
        return -np.sum(probs * np.log(probs))

    info_gain     = entropy(prev_belief) - entropy(curr_belief)
    shaped_reward = base_reward + 0.05 * info_gain
    return float(shaped_reward)


def reveal_info_penalty(state, chosen_card, all_cards, env):
    """
    Optional separate reward signal for the reveal head.

    Penalises revealing a card that the suggester didn't already know
    the RL agent held (i.e. a high-information reveal).
    Rewards revealing a card that was already well-known.

    This can be used instead of — or compared against — the win/loss signal
    for the reveal head. To use it, call it from the train loop and pass the
    result to agent.store_reveal() as the reward.

    Returns a float in roughly [-0.1, +0.1].
    """
    rl_player = env.players[env.rl_player_index]

    # Find the suggester from the most recent log entry
    public_log = env.get_suggestion_history()
    if not public_log:
        return 0.0

    last_rec  = public_log[-1]
    suggester = last_rec.get("suggester")
    if suggester is None or suggester is rl_player:
        return 0.0

    # How much did the suggester already believe the RL agent had this card?
    # Higher belief => revealing it costs less information
    prob_already_known = suggester.ownersAndCards.get(rl_player, {}).get(chosen_card, 0.0)

    # +0.1 if completely expected (prob=1), -0.1 if totally surprising (prob=0)
    return float(0.1 * (2 * prob_already_known - 1))
