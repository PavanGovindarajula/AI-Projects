"""Educational, dimensionless fed-batch simulator and tabular Q controller."""
from dataclasses import dataclass
import random

ACTIONS = (0.0, 0.5, 1.0, 1.5, 2.0)

@dataclass
class State:
    day: int = 0
    cells: float = 1.0
    glucose: float = 2.0
    lactate: float = 0.0
    product: float = 0.0
    ph: float = 7.0
    oxygen: float = 50.0
    temperature: float = 37.0

class Reactor:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.state = State(glucose=self.rng.uniform(1.7, 2.3))

    def step(self, feed):
        s = self.state
        # Dimensionless toy dynamics, not a model of a real cell line.
        available = s.glucose + feed
        uptake = min(available, 0.22 * s.cells)
        excess = max(0.0, available - 3.0)
        s.lactate = max(0.0, s.lactate + 0.22 * excess - 0.03)
        s.glucose = max(0.0, available - uptake)
        s.oxygen = max(0.0, 55 - 2.5 * s.cells + self.rng.gauss(0, 0.3))
        s.ph = 7.05 - 0.07 * s.lactate
        quality = min(1.0, uptake / max(0.001, 0.18 * s.cells))
        quality *= max(0.1, 1 - 0.10 * s.lactate)
        quality *= min(1.0, s.oxygen / 30)
        s.product += 0.16 * s.cells * quality
        s.cells *= 1 + 0.16 * quality - 0.025
        s.day += 1
        return s, s.day >= 14

def features(s):
    return (min(s.day // 3, 4), min(int(s.glucose), 5), min(int(s.lactate), 4), min(int(s.cells // 2), 4))

def train(episodes=12000, seed=42):
    rng = random.Random(seed)
    q = {}
    for episode in range(episodes):
        env = Reactor(rng.randrange(10**9))
        epsilon = max(0.05, 0.8 * (1 - episode / episodes))
        while True:
            key = features(env.state)
            values = q.setdefault(key, [0.0] * len(ACTIONS))
            action = rng.randrange(len(ACTIONS)) if rng.random() < epsilon else max(range(len(ACTIONS)), key=values.__getitem__)
            before = env.state.product
            state, done = env.step(ACTIONS[action])
            # Incremental product is reward shaping; total reward equals final product.
            reward = state.product - before
            future = 0 if done else max(q.setdefault(features(state), [0.0] * len(ACTIONS)))
            values[action] += 0.15 * (reward + future - values[action])
            if done: break
    return q

def policy(q, state):
    values = q.get(features(state), [0.0] * len(ACTIONS))
    return ACTIONS[max(range(len(ACTIONS)), key=values.__getitem__)]
