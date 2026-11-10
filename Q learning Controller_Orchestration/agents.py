"""Explicit orchestration with deterministic safety and review gates."""
from dataclasses import asdict
import math
from .observability import new_trace_id
from .core import ACTIONS, Reactor, policy

class SafetyGuardian:
    def check(self, state, proposed):
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in asdict(state).values()):
            return False, 'Invalid or non-finite state; pause for human review.'
        if isinstance(proposed, bool) or not isinstance(proposed, (int, float)) or not math.isfinite(proposed) or proposed not in ACTIONS:
            return False, 'Feed is outside the allowed discrete action set.'
        if not 6.8 <= state.ph <= 7.2 or state.oxygen < 30 or not 36 <= state.temperature <= 38:
            return False, 'Observed state outside safety envelope; pause for human review.'
        if state.glucose + proposed > 5:
            return False, 'Proposed feed exceeds the glucose guardrail; pause for human review.'
        return True, 'Within simulated guardrails.'

class Monitor:
    def assess(self, s):
        alerts = []
        if s.glucose < 0.5: alerts.append('Low glucose')
        if s.lactate > 2: alerts.append('Elevated lactate')
        if s.oxygen < 35: alerts.append('Oxygen approaching limit')
        return alerts

class Diagnostics:
    def diagnose(self, alerts):
        knowledge = {'Low glucose': 'Review feed history and substrate consumption.', 'Elevated lactate': 'Review prior glucose excess and metabolic trends.', 'Oxygen approaching limit': 'Review oxygen demand and sensor quality.'}
        return [knowledge[a] for a in alerts]

class Orchestrator:
    def __init__(self, q, event_sink=None):
        self.q, self.guardian, self.monitor, self.diagnostics = q, SafetyGuardian(), Monitor(), Diagnostics()
        self.event_sink = event_sink

    def run(self, seed, mode='rl'):
        if mode not in ('rl', 'fixed'):
            raise ValueError('mode must be rl or fixed')
        trace_id = new_trace_id()
        def emit(event, **fields):
            if self.event_sink is not None:
                self.event_sink.emit(trace_id, event, **fields)
        env, records = Reactor(seed), []
        emit('batch_started', mode=mode, seed=seed)
        while env.state.day < 14:
            s = env.state
            proposed = policy(self.q, s) if mode == 'rl' else 0.5
            alerts = self.monitor.assess(s)
            allowed, reason = self.guardian.check(s, proposed)
            row = {'before': asdict(s), 'proposed_feed': proposed, 'alerts': alerts, 'diagnosis': self.diagnostics.diagnose(alerts), 'safety': reason}
            records.append(row)
            emit('safety_decision', day=s.day, proposed_feed=proposed, allowed=allowed, reason=reason, alerts=alerts)
            if not allowed:
                emit('batch_paused', day=s.day, reason=reason)
                return {'trace_id': trace_id, 'status': 'awaiting_human_review', 'records': records, 'final': asdict(s)}
            state, _ = env.step(proposed)
            row['after'] = asdict(state)
            emit('step_completed', day=state.day, product=state.product)
        emit('batch_completed', day=env.state.day, product=env.state.product)
        return {'trace_id': trace_id, 'status': 'completed', 'records': records, 'final': asdict(env.state)}

def report(run):
    return {'trace_id': run['trace_id'], 'status': run['status'], 'days_completed': run['final']['day'], 'product': round(run['final']['product'], 3), 'alerts': sum(len(r['alerts']) for r in run['records']), 'batch_record': run['records']}
