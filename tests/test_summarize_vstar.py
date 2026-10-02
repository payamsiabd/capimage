import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import summarize_vstar as sv  # noqa: E402


@pytest.mark.parametrize('prediction, options, expected', [
    ('... so the answer is \\boxed{B}', {'A': 'white', 'B': 'red'}, 'B'),
    ('\\boxed{(A)}', {'A': 'white', 'B': 'red'}, 'A'),
    ('\\boxed{B. red}', {'A': 'white', 'B': 'red'}, 'B'),
    ('\\boxed{\\text{B}}', {'A': 'white', 'B': 'red'}, 'B'),
    ('\\boxed{Red.}', {'A': 'white', 'B': 'red'}, 'B'),
    ('\\boxed{A} first, then \\boxed{B}', {'A': 'white', 'B': 'red'}, 'B'),
    ('\\boxed{C}', {'A': 'left', 'B': 'right'}, None),
    ('\\boxed{Blue}', {'A': 'white', 'B': 'red'}, None),
    ('The answer is B.', {'A': 'white', 'B': 'red'}, None),
    ('\\boxed{B', {'A': 'white', 'B': 'red'}, None),
])
def test_boxed_choice(prediction, options, expected):
    assert sv.boxed_choice(prediction, options) == expected


def test_category_group():
    assert sv.category_group('direct_attributes') == 'Attribute'
    assert sv.category_group('relative_position') == 'Spatial'
    with pytest.raises(ValueError):
        sv.category_group('counting')


def make_results(n_attr_correct=101, n_spatial_correct=63):
    """Synthetic VLMEvalKit *_result table with V*'s 115 attribute + 76 spatial questions."""
    rows = []
    for i in range(115):
        hit = int(i < n_attr_correct)
        rows.append(dict(index=i, question='What is the color of the cup?', A='white', B='red', C='blue', D='black',
                         answer='A', category='direct_attributes', prediction=f'\\boxed{{{"A" if hit else "B"}}}',
                         hit=hit, log='Match Log: A. '))
    for i in range(76):
        hit = int(i < n_spatial_correct)
        rows.append(dict(index=115 + i, question='Is the dog on the left or right of the cat?', A='left',
                         B='right', C=None, D=None, answer='B', category='relative_position',
                         prediction='I think it is on the right side.', hit=hit, log='Match Log: B. '))
    return pd.DataFrame(rows)


def test_summarize_reproduces_paper_rounding():
    summary = sv.summarize(make_results())
    paper = sv.PAPER['CapImagine-7B']
    for name in sv.SPLITS:
        assert round(summary['splits'][name]['acc'], 1) == paper[name]
    assert summary['splits']['Overall']['correct'] == 164
    # Spatial predictions carry no \boxed{}, so the diagnostic column only credits attribute hits.
    assert summary['splits']['Attribute']['boxed_correct'] == 101
    assert summary['splits']['Spatial']['boxed_correct'] == 0
    assert summary['n_no_boxed'] == 76


def test_summarize_flags_judge_fallbacks():
    df = make_results()
    df.loc[:9, 'log'] = 'Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.'
    df.loc[10:12, 'log'] = 'Failed to predict, thus randomly generate one. '
    summary = sv.summarize(df)
    assert summary['n_exact_match_fallback'] == 10
    assert summary['n_random_fallback'] == 3
    report = sv.format_report(summary, 'CapImagine-7B', 'x/T20260301_G0bac5c06/r.xlsx', 'chatgpt-0125')
    assert 'not the paper protocol' in report
    assert 'randomly' in report or 'random option' in report


def test_main_finds_latest_vlmevalkit_result(tmp_path):
    model = 'CapImagine-7B'
    old = tmp_path / model / 'T20260301_G0bac5c06'
    new = tmp_path / model / 'T20260302_G0bac5c06'
    for d in (old, new):
        d.mkdir(parents=True)
    make_results(0, 0).to_excel(old / f'{model}_VStarBench_openai_result.xlsx', index=False)
    make_results().to_excel(new / f'{model}_VStarBench_openai_result.xlsx', index=False)

    summary = sv.main(['--work-dir', str(tmp_path), '--model-name', model, '--judge', 'chatgpt-0125'])

    assert summary['result_file'].startswith(str(new))
    assert round(summary['splits']['Overall']['acc'], 1) == 85.9
    with open(new / f'{model}_VStarBench_openai_result_summary.json') as f:
        assert json.load(f)['splits']['Spatial']['correct'] == 63


def test_custom_model_compares_against_reference(tmp_path):
    run_dir = tmp_path / 'my-lora' / 'T20261002_G0bac5c06'
    run_dir.mkdir(parents=True)
    make_results().to_excel(run_dir / 'my-lora_VStarBench_openai_result.xlsx', index=False)
    summary = sv.main(['--work-dir', str(tmp_path), '--model-name', 'my-lora', '--reference', 'CapImagine-7B'])
    assert summary['paper'] == sv.PAPER['CapImagine-7B']
    report = sv.format_report(summary, 'my-lora', summary['result_file'], 'chatgpt-0125', 'CapImagine-7B')
    assert 'paper (CapImagine-7B)' in report and '| Overall | 191 | 164 | 85.9 | 85.9 | +0.0 |' in report


def test_main_exits_when_no_result(tmp_path):
    with pytest.raises(SystemExit):
        sv.main(['--work-dir', str(tmp_path)])
