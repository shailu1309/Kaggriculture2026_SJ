"""Local test harness + episode logger.

Two jobs:
  1. Score main.py against the built-in agents so we have a baseline number
     that every future version gets compared against.
  2. Log every (observation, action) pair to JSONL. This is the training data
     for the later ML phases -- it costs nothing to collect now and cannot be
     recovered retroactively, so it runs from day one.

Usage:
    python run_local.py                     # 3 games vs random
    python run_local.py --opponent starter --games 5
    python run_local.py --check-directions  # verify NORTH/SOUTH mapping
"""

import argparse
import json
import os
import time

from kaggle_environments import make

LOG_DIR = "logs"


def wrap_with_logging(agent_fn, sink):
    """Record what the agent saw and what it did, every turn."""
    def wrapped(obs, config=None):
        action = agent_fn(obs)
        sink.append({
            "day": obs.get("day"),
            "hour": obs.get("hour"),
            "step": obs.get("step"),
            "money": obs["farms"][obs["player"]]["money"],
            "market_prices": dict(obs["market"]["prices"]),
            "shed": dict(obs["private"]["shed"]),
            "action": action,
        })
        return action
    return wrapped


def preflight(opponent="random"):
    """Load main.py by path, exactly as Kaggle does.

    Kaggle resolves the *last callable defined in the file* as the agent, so a
    helper defined below agent() silently becomes the submission. Importing
    `agent` by name (as the scored runs below do) hides that entirely -- which
    is how a broken v1 passed local testing and errored on submit. Never skip
    this before submitting.
    """
    env = make("kaggriculture", debug=True)
    env.run(["main.py", opponent])
    s = env.steps[-1][0]
    ok = s.status == "DONE"
    print(f"preflight (loaded from main.py): status={s.status} reward={s.reward}")
    if not ok:
        print("  !! Kaggle would reject this. Is `agent` the LAST function in main.py?")
    return ok


def run(games, opponent, log):
    from main import agent as my_agent

    if not preflight(opponent):
        print("\nAborting scored runs -- fix the preflight failure first.")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    results = []

    for g in range(games):
        turns = []
        contender = wrap_with_logging(my_agent, turns) if log else my_agent

        env = make("kaggriculture", debug=True)
        env.run([contender, opponent])

        final = env.steps[-1]
        mine, theirs = final[0].reward, final[1].reward
        status = final[0].status
        results.append((mine, theirs, status))
        print(f"game {g + 1}: me={mine}  {opponent}={theirs}  status={status}")

        if status != "DONE":
            print("  !! agent did not finish cleanly -- check the traceback above")

        if log:
            path = os.path.join(LOG_DIR, f"ep_{int(time.time())}_{g}.jsonl")
            with open(path, "w") as f:
                for row in turns:
                    f.write(json.dumps(row) + "\n")

    wins = sum(1 for m, t, _ in results if m > t)
    scores = [m for m, _, _ in results if isinstance(m, (int, float))]
    print(f"\n{wins}/{games} wins vs {opponent}")
    if scores:
        print(f"mean final money: {sum(scores) / len(scores):.0f}")
        print(f"range: {min(scores):.0f} - {max(scores):.0f}")


def check_directions():
    """main.py assumes NORTH decreases y. Confirm that against the real env."""
    def prober(obs):
        if obs.get("step", 0) == 0:
            prober.start = tuple(obs["farms"][obs["player"]]["farmer"])
            return {"farmer": ["NORTH"], "hands": [], "market": []}
        if obs.get("step", 0) == 1:
            prober.after = tuple(obs["farms"][obs["player"]]["farmer"])
        return {"farmer": ["PASS"], "hands": [], "market": []}

    env = make("kaggriculture", configuration={"episodeSteps": 4}, debug=True)
    env.run([prober, "random"])
    sx, sy = prober.start
    ax, ay = getattr(prober, "after", prober.start)
    print(f"farmer {prober.start} --NORTH--> {(ax, ay)}")
    if ay < sy:
        print("OK: NORTH decreases y, main.py's assumption holds")
    elif ay > sy:
        print("MISMATCH: NORTH increases y -- swap NORTH/SOUTH in step_toward()")
    else:
        print("no movement (edge of board?) -- rerun from a non-edge tile")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--opponent", default="random", choices=["random", "starter", "pass"])
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--check-directions", action="store_true")
    a = p.parse_args()

    if a.check_directions:
        check_directions()
    else:
        run(a.games, a.opponent, log=not a.no_log)
