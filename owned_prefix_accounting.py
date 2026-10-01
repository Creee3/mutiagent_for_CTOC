"""Content-only counterfactual accounting from the finished free-chain ledger.

This does not reconstruct caches, run models, or estimate network latency.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(summary):
    assert summary['run_name'] == 'freechain_20261001_125456'
    assert summary['main_questions'] == len(summary['cases']) == 3
    assert summary['structural_assertions_passed'] is True
    rows = []
    for item in summary['cases']:
        row = {'id': item['id']}
        for representation in ('kv', 'text'):
            observed = item['repeated'][representation]
            a = observed['A_history_bytes_in_B_to_C']
            linear = observed['total_content_bytes']
            assert type(a) is int and type(linear) is int
            assert 0 < 2*a < linear
            ab = linear-a
            b = ab-a
            full_edges = {'A_to_B': a, 'A_to_C': a, 'B_to_C': ab}
            delta_edges = {'A_to_B': a, 'A_to_C': a, 'B_to_C': b}
            full_total, delta_total = sum(full_edges.values()), sum(delta_edges.values())
            assert full_total == 3*a+b and delta_total == 2*a+b == linear
            row[representation] = {
                'A_content_bytes': a, 'B_added_content_bytes': b,
                'observed_linear_full_content_bytes': linear,
                'fixed_triangle_full_edges': full_edges,
                'fixed_triangle_delta_edges': delta_edges,
                'fixed_triangle_full_content_bytes': full_total,
                'fixed_triangle_delta_content_bytes': delta_total,
                'saved_content_bytes_on_same_triangle': full_total-delta_total,
                'saved_fraction_on_same_triangle': a/full_total,
                'saved_fraction_on_B_to_C_only': a/ab,
                'saved_content_bytes_vs_observed_linear': linear-delta_total,
            }
        rows.append(row)
    totals = {}
    for representation in ('kv', 'text'):
        full = sum(x[representation]['fixed_triangle_full_content_bytes'] for x in rows)
        delta = sum(x[representation]['fixed_triangle_delta_content_bytes'] for x in rows)
        linear = sum(x[representation]['observed_linear_full_content_bytes'] for x in rows)
        totals[representation] = {
            'fixed_triangle_full_content_bytes': full,
            'fixed_triangle_delta_content_bytes': delta,
            'observed_linear_full_content_bytes': linear,
            'saved_fraction_on_same_triangle': (full-delta)/full,
            'saved_content_bytes_vs_observed_linear': linear-delta,
        }
        assert linear == delta
    return {
        'cases': rows, 'totals': totals,
        'scope': 'Counterfactual content-only arithmetic; no new transmission or model execution.',
        'representations': {'kv': 'Raw KV tensor bytes only; excludes wire history tokens.',
                            'text': 'Exact history int32 token bytes, not UTF-8 text bytes.'},
        'excludes': ['packet headers', 'references', 'metadata', 'receiver storage and restore work',
                    'network and serialization latency'],
        'new_model_calls': 0,
        'important_boundary': 'Adding A_to_C just to create reuse does not save content over the observed linear chain.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='runs/freechain_20261001_125456/summary.json')
    parser.add_argument('--output', default='checks/owned_prefix_accounting.json')
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    result = calculate(json.loads(source.read_text(encoding='utf-8')))
    result['source'] = {'path': source.as_posix(), 'sha256': digest(source)}
    result['script_sha256'] = digest(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['totals'], ensure_ascii=False))


if __name__ == '__main__':
    main()
