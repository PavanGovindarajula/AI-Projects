import unittest
from bioreactor_rl.core import Reactor, State, train
from bioreactor_rl.agents import SafetyGuardian, Orchestrator

class ProjectTests(unittest.TestCase):
    def test_episode_and_reproducibility(self):
        a, b = Reactor(7), Reactor(7)
        for _ in range(14):
            a.step(0.5); b.step(0.5)
        self.assertEqual(a.state, b.state)
        self.assertEqual(a.state.day, 14)
        self.assertGreater(a.state.product, 0)
    def test_guardian_blocks_unsafe(self):
        guard = SafetyGuardian()
        self.assertFalse(guard.check(State(ph=6.7), 0.5)[0])
        self.assertFalse(guard.check(State(glucose=5), 1.0)[0])
        self.assertFalse(guard.check(State(), 99)[0])
    def test_training_and_report(self):
        run = Orchestrator(train(100, 1)).run(1)
        self.assertIn(run['status'], ('completed', 'awaiting_human_review'))
        self.assertTrue(run['records'])

if __name__ == '__main__': unittest.main()
