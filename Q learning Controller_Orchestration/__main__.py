import argparse
import json
import statistics
from .core import train
from .agents import Orchestrator, report
from .observability import JsonlEventSink

def main():
    parser = argparse.ArgumentParser(description='Run a toy fed-batch RL experiment')
    parser.add_argument('--episodes', type=int, default=12000)
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--output', default='batch_report.json')
    parser.add_argument('--audit-log', help='Append structured JSON events to this file')
    args = parser.parse_args()
    if args.episodes < 1 or args.batches < 1: parser.error('episodes and batches must be positive')
    orchestrator = Orchestrator(train(args.episodes), JsonlEventSink(args.audit_log) if args.audit_log else None)
    results = {mode: [orchestrator.run(100000 + i, mode) for i in range(args.batches)] for mode in ('fixed', 'rl')}
    summary = {}
    for mode, runs in results.items():
        completed = [r['final']['product'] for r in runs if r['status'] == 'completed']
        summary[mode] = {'completed': len(completed), 'paused': len(runs)-len(completed), 'mean_completed_product': round(statistics.mean(completed), 3) if completed else None}
    with open(args.output, 'w') as f: json.dump({'summary': summary, 'example_rl_batch': report(results['rl'][0])}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print('Report:', args.output)

if __name__ == '__main__': main()
