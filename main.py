"""Kaggriculture v1 — crops-only heuristic agent.

Design (deliberately simple; see strategy.md for the roadmap):
  - Stateless: every decision is recomputed from the observation each turn.
    No cross-turn memory means no state-sync bugs and no desync on Kaggle.
  - Each unit (farmer + hired hands) is assigned the nearest unclaimed task.
    Task priority: WATER (plants die without it) > HARVEST > PLANT > DIG weed.
  - Crop choice is scored on live market price, so the agent rotates away from
    crops whose price it has crashed by selling.
  - Sells are drip-fed in small chunks; a single big SELL order walks its own
    price down, which is how you turn $250 melons into $1 melons.
  - No animals, no fertilizer in v1. Those are the next increments.
"""

import math

# ---------------------------------------------------------------- crop tuning

# harvest_age: age (in days) at which yield stops improving, so harvesting
# earlier throws away units. Values come from the README yield rules:
#   one-time crops gain +1 unit per watered day in the bonus window, which
#   starts at ceil(max_yield_day / 2).
CROPS = {
    "WHEAT":  {"seed": 10,  "base": 25,  "first_yield": 2,  "harvest_age": 4,  "max_units": 4},
    "CARROT": {"seed": 20,  "base": 35,  "first_yield": 2,  "harvest_age": 3,  "max_units": 3},
    "MELON":  {"seed": 80,  "base": 250, "first_yield": 10, "harvest_age": 10, "max_units": 6},
}

SEASON_DAYS = 30
SHED_CAP = 100
MAX_MARKET_ORDERS = 10
MAX_HANDS = 6          # fib cost is 1,1,2,3,5,8 -> $20/day total. Noise.
LAND_COSTS = [1000, 2000, 4000]
QUADRANT_ORDER = ["NE", "SW", "SE"]


def step_toward(ux, uy, tx, ty):
    """One orthogonal step from (ux,uy) toward (tx,ty)."""
    if tx > ux:
        return "EAST"
    if tx < ux:
        return "WEST"
    if ty > uy:
        return "SOUTH"
    if ty < uy:
        return "NORTH"
    return "PASS"


def crop_score(crop, price, days_left):
    """Expected profit per tile-day at the current market price.

    Dividing by days occupied is what stops melon (huge total, slow) from
    automatically beating wheat (small total, fast) when the season is short.
    """
    c = CROPS[crop]
    days = c["harvest_age"]
    if days_left < days + 1:
        return -1.0                      # cannot mature before the season ends
    revenue = c["max_units"] * price
    return (revenue - c["seed"]) / days


def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    priv = obs["private"]
    day, hour = obs["day"], obs["hour"]
    tiles = me["tiles"]
    money = me["money"]
    prices = obs["market"]["prices"]
    shed = priv["shed"]
    seeds = dict(priv["seeds"])

    size = len(tiles)
    half = size // 2
    days_left = SEASON_DAYS - day
    endgame = days_left <= 2

    # ------------------------------------------------------------ crop choice
    ranked = sorted(
        CROPS,
        key=lambda c: crop_score(c, prices.get(c, CROPS[c]["base"]), days_left),
        reverse=True,
    )
    best_crop = next(
        (c for c in ranked
         if crop_score(c, prices.get(c, CROPS[c]["base"]), days_left) > 0),
        None,
    )

    # ----------------------------------------------------------- build tasks
    # (priority, x, y, action). Lower priority number wins.
    tasks = []
    empty_tiles = 0
    for y in range(size):
        for x in range(size):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            if t is None:
                empty_tiles += 1
                # Never plant so late in the day that the seed cannot be
                # watered before the end-of-day refresh -- an unwatered fresh
                # planting is a weed by morning.
                if best_crop and hour < 22 and not endgame:
                    tasks.append((2, x, y, ["PLANT", best_crop]))
                continue
            kind = t.get("kind")
            if kind == "WEED":
                tasks.append((3, x, y, ["DIG"]))
            elif kind == "PLANT":
                crop = t["crop"]
                cfg = CROPS.get(crop)
                age = day - t["planted_day"]
                ready = t.get("yield_units", 0) > 0
                ripe = cfg and age >= cfg["harvest_age"]
                if ready and (ripe or endgame):
                    tasks.append((1, x, y, ["HARVEST"]))
                elif not t.get("watered_today"):
                    tasks.append((0, x, y, ["WATER"]))

    # --------------------------------------------------------- assign units
    units = [tuple(me["farmer"])] + [tuple(h) for h in me["hands"]]
    claimed = set()
    planned_plants = {}
    unit_actions = []

    for (ux, uy) in units:
        candidates = [
            t for t in tasks
            if (t[1], t[2]) not in claimed
            and not (t[3][0] == "PLANT"
                     and planned_plants.get(t[3][1], 0) >= seeds.get(t[3][1], 0))
        ]
        if not candidates:
            unit_actions.append(["PASS"])
            continue

        # Nearest task, tie-broken by priority.
        prio, tx, ty, action = min(
            candidates,
            key=lambda t: (abs(t[1] - ux) + abs(t[2] - uy), t[0]),
        )
        claimed.add((tx, ty))
        if action[0] == "PLANT":
            planned_plants[action[1]] = planned_plants.get(action[1], 0) + 1

        if (ux, uy) == (tx, ty):
            unit_actions.append(action)
        else:
            unit_actions.append([step_toward(ux, uy, tx, ty)])

    farmer_action = unit_actions[0] if unit_actions else ["PASS"]
    hand_actions = unit_actions[1:]

    # -------------------------------------------------------- market orders
    market = []
    budget = money

    # 1. Sell. Drip-feed so we don't walk our own price into the floor.
    shed_total = sum(v for k, v in shed.items() if k in prices)
    for item, qty in sorted(shed.items(), key=lambda kv: -prices.get(kv[0], 0)):
        if qty <= 0 or item not in prices or len(market) >= MAX_MARKET_ORDERS:
            continue
        price = prices[item]
        base = CROPS.get(item, {}).get("base", price)
        crowded = shed_total > 0.7 * SHED_CAP
        if endgame or crowded:
            n = qty                       # dump: unsold stock scores nothing
        elif price >= 0.6 * base:
            n = max(1, qty // 8)
        else:
            continue                      # price is depressed, wait it out
        market.append(["SELL", item, int(n)])

    # 2. Hire at the start of the day. Cost is fib(n) -- pennies against the
    #    throughput a extra pair of hands buys.
    if hour == 0 and not endgame:
        want = min(MAX_HANDS, max(1, math.ceil(len(tasks) / 4)))
        for i in range(want):
            if len(market) >= MAX_MARKET_ORDERS:
                break
            market.append(["HIRE"])

    # 3. Seeds. Keep a shallow buffer so PLANT is never starved.
    if best_crop and not endgame and len(market) < MAX_MARKET_ORDERS:
        cfg = CROPS[best_crop]
        want = min(empty_tiles, 10) - seeds.get(best_crop, 0)
        affordable = int(budget * 0.5 // cfg["seed"])
        n = max(0, min(want, affordable))
        if n > 0:
            market.append(["BUY_SEED", best_crop, n])
            budget -= n * cfg["seed"]

    # 4. Land. Only worth it with enough season left to farm the new tiles,
    #    and only out of surplus so it never starves the seed budget.
    bought = len(me["unlocked_quadrants"]) - 1
    if (bought < len(LAND_COSTS) and days_left > 8
            and empty_tiles <= 2 and len(market) < MAX_MARKET_ORDERS):
        cost = LAND_COSTS[bought]
        if budget > cost * 2:
            market.append(["BUY_LAND"])

    return {
        "farmer": farmer_action,
        "hands": hand_actions,
        "market": market[:MAX_MARKET_ORDERS],
    }
