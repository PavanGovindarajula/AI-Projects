import json
import os
import tempfile
import unittest
from unittest.mock import patch

from bioreactor_rl.agents import Orchestrator, SafetyGuardian
from bioreactor_rl.core import State
from bioreactor_rl.observability import JsonlEventSink


class ObservabilityTests(unittest.TestCase):
    def test_audit_events_correlate_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'audit.jsonl')
            run = Orchestrator({}, JsonlEventSink(path)).run(4, 'fixed')
            with open(path, encoding='utf-8') as stream:
                events = [json.loads(line) for line in stream]
            self.assertEqual(events[0]['event'], 'batch_started')
            self.assertEqual(events[-1]['event'], 'batch_completed')
            self.assertTrue(all(e['trace_id'] == run['trace_id'] for e in events))
            self.assertEqual(len({e['event_id'] for e in events}), len(events))
            self.assertEqual(sum(e['event'] == 'safety_decision' for e in events), 14)
            self.assertEqual(os.stat(path).st_mode & 0o077, 0)

    def test_pause_never_steps_and_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'audit.jsonl')
            with patch.object(SafetyGuardian, 'check', return_value=(False, 'blocked')):
                run = Orchestrator({}, JsonlEventSink(path)).run(3, 'fixed')
            self.assertEqual(run['status'], 'awaiting_human_review')
            self.assertEqual(run['final']['day'], 0)
            with open(path, encoding='utf-8') as stream:
                events = [json.loads(line)['event'] for line in stream]
            self.assertEqual(events, ['batch_started', 'safety_decision', 'batch_paused'])

    def test_invalid_inputs_fail_closed(self):
        guardian = SafetyGuardian()
        for state, action in [(State(ph=float('nan')), 0.5), (State(oxygen=float('inf')), 0.5), (State(), True), (State(), float('nan'))]:
            self.assertFalse(guardian.check(state, action)[0])
        with self.assertRaises(ValueError):
            Orchestrator({}).run(1, 'unknown')

    def test_log_failure_stops_batch(self):
        class BrokenSink:
            def emit(self, *args, **kwargs):
                raise OSError('audit unavailable')
        with self.assertRaises(OSError):
            Orchestrator({}, BrokenSink()).run(1, 'fixed')

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW'), 'requires O_NOFOLLOW')
    def test_audit_log_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, 'target')
            path = os.path.join(directory, 'link')
            os.symlink(target, path)
            with self.assertRaises(OSError):
                JsonlEventSink(path).emit('trace', 'event')
            self.assertFalse(os.path.exists(target))
