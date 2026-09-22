"""Summarise a VLMEvalKit V* Bench run and compare it with the CapImagine paper.

Reads the per-question judge result VLMEvalKit writes to
``<work_dir>/<model>/T<date>_G<commit>/<model>_VStarBench_<judge>_result.xlsx``
and reports Attribute / Spatial / Overall accuracy next to Table 1 of the paper.

It also reports a judge-free diagnostic: accuracy when the answer is read
strictly from the last ``\\boxed{}`` in the model output. That is NOT the
paper's protocol; it only shows how much the LLM judge changes the score.
"""
import argparse
import glob
import json
import os
import re
import string
import sys

import pandas as pd

# Table 1 of "Imagination Helps Visual Reasoning, But Not Yet in Latent Space" (ICML 2026).
PAPER = {
    'CapImagine-7B': {'Attribute': 87.8, 'Spatial': 82.9, 'Overall': 85.9},
    'Qwen2.5-VL-7B-Instruct': {'Attribute': 77.4, 'Spatial': 75.0, 'Overall': 76.4},
}
EXPECTED_COUNTS = {'Attribute': 115, 'Spatial': 76, 'Overall': 191}
SPLITS = ['Attribute', 'Spatial', 'Overall']
# VLMEvalKit names the result file after the judge, using these aliases.
JUDGE_TAGS = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
# Match-log fragments VLMEvalKit writes when the judge was not (successfully) used.
EXACT_MATCH_LOG = 'no GPT-based answer matching'
RANDOM_LOG = 'Failed to predict, thus randomly generate one'


def category_group(category):
    c = str(category).lower()
    if 'attr' in c:
        return 'Attribute'
    if 'position' in c or 'spatial' in c or 'relative' in c:
        return 'Spatial'
    raise ValueError(f'Unrecognised V* category: {category!r}')


def last_boxed(text):
    """Content of the last brace-balanced ``\\boxed{...}`` in `text`, or None."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    begin = start + len('\\boxed{')
    depth = 1
    for i in range(begin, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[begin:i]
    return None


def _norm(s):
    return re.sub(r'\s+', ' ', str(s)).strip().strip('.').strip().lower()


def boxed_choice(prediction, options):
    """Option letter named by the last ``\\boxed{}`` in `prediction`, or None."""
    content = last_boxed(str(prediction))
    if content is None:
        return None
    content = re.sub(r'\\text\{(.*?)\}', r'\1', content).strip()
    by_text = [k for k, v in options.items() if _norm(v) == _norm(content)]
    if len(by_text) == 1:
        return by_text[0]
    m = re.match(r'\(?([A-Z])\)?(?=$|[\s.:)])', content)
    if m and m.group(1) in options:
        return m.group(1)
    return None


def summarize(df):
    df = df.copy()
    df['group'] = df['category'].map(category_group)
    letters = [c for c in string.ascii_uppercase if c in df.columns]
    options = [{c: row[c] for c in letters if pd.notna(row[c])} for _, row in df.iterrows()]
    df['boxed'] = [boxed_choice(p, o) for p, o in zip(df['prediction'], options)]
    df['boxed_hit'] = (df['boxed'] == df['answer']).astype(int)

    splits = {}
    for name in SPLITS:
        sub = df if name == 'Overall' else df[df['group'] == name]
        splits[name] = {
            'n': len(sub),
            'correct': int(sub['hit'].sum()),
            'acc': 100 * sub['hit'].mean() if len(sub) else float('nan'),
            'boxed_correct': int(sub['boxed_hit'].sum()),
            'boxed_acc': 100 * sub['boxed_hit'].mean() if len(sub) else float('nan'),
        }
    logs = df['log'].astype(str)
    return {
        'splits': splits,
        'n_no_boxed': int(df['boxed'].isna().sum()),
        'n_exact_match_fallback': int(logs.str.contains(EXACT_MATCH_LOG, regex=False).sum()),
        'n_random_fallback': int(logs.str.contains(RANDOM_LOG, regex=False).sum()),
    }


def find_result_file(work_dir, model_name, judge):
    tag = JUDGE_TAGS.get(judge, judge)
    pattern = os.path.join(work_dir, model_name, 'T*_G*', f'{model_name}_VStarBench_{tag}_result.xlsx')
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f'No VLMEvalKit result file matches {pattern}')
    return files[-1]


def format_report(summary, model_name, result_file, judge):
    paper = PAPER.get(model_name)
    run_dir = os.path.basename(os.path.dirname(result_file))
    lines = [
        f'V* Bench | {model_name} | judge: {judge} | run: {run_dir}',
        f'result file: {result_file}',
        '',
        '| Split | n | correct | ours | paper | diff | boxed-only (diagnostic) |',
        '|---|---|---|---|---|---|---|',
    ]
    for name in SPLITS:
        s = summary['splits'][name]
        ref = f'{paper[name]:.1f}' if paper else '-'
        diff = f'{s["acc"] - paper[name]:+.1f}' if paper else '-'
        lines.append(
            f'| {name} | {s["n"]} | {s["correct"]} | {s["acc"]:.1f} | {ref} | {diff} | {s["boxed_acc"]:.1f} |'
        )

    warnings = []
    for name, expected in EXPECTED_COUNTS.items():
        if summary['splits'][name]['n'] != expected:
            warnings.append(f'{name} has {summary["splits"][name]["n"]} questions; V* has {expected}.')
    if summary['n_exact_match_fallback']:
        warnings.append(
            f'{summary["n_exact_match_fallback"]} answers were scored by exact matching only: the judge API was '
            'not working, so this is not the paper protocol. Fix the judge and re-run with --mode eval.'
        )
    if summary['n_random_fallback']:
        warnings.append(
            f'The judge failed on {summary["n_random_fallback"]} answers and VLMEvalKit substituted a random option.'
        )
    if summary['n_no_boxed']:
        warnings.append(f'{summary["n_no_boxed"]} outputs have no parseable \\boxed{{}} answer (diagnostic column only).')
    if warnings:
        lines += [''] + [f'WARNING: {w}' for w in warnings]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work-dir', default='outputs')
    parser.add_argument('--model-name', default='CapImagine-7B')
    parser.add_argument('--judge', default='chatgpt-0125')
    parser.add_argument('--result-file', help='explicit *_result.xlsx (skips the search in --work-dir)')
    args = parser.parse_args(argv)

    result_file = args.result_file or find_result_file(args.work_dir, args.model_name, args.judge)
    summary = summarize(pd.read_excel(result_file))
    print(format_report(summary, args.model_name, result_file, args.judge))

    summary.update(model=args.model_name, judge=args.judge, result_file=result_file,
                   paper=PAPER.get(args.model_name))
    out = os.path.splitext(result_file)[0] + '_summary.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nsummary written to {out}')
    return summary


if __name__ == '__main__':
    main()
