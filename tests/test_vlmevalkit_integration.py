"""Run VLMEvalKit's own V* scoring code on synthetic outputs and summarise the file it writes.

Needs the pinned VLMEvalKit installed (scripts/setup_env.sh) but no GPU, model or network:
the dataset is synthetic and the judge is a stub.
"""
import os
import sys

import pandas as pd
import pytest

vlmeval = pytest.importorskip('vlmeval')
from vlmeval.dataset import ImageMCQDataset  # noqa: E402
from vlmeval.smp import dump  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import summarize_vstar as sv  # noqa: E402

REASONING = ('Suppose a red rectangle had been marked on the upper part of the image, highlighting the cup. '
             'A closer look shows the object clearly. ')


def synthetic_vstar():
    """Meta table shaped like VStarBench.tsv (without images) plus CapImagine-style predictions."""
    rows, preds = [], []
    for i in range(4):
        rows.append(dict(index=i, question='What is the color of the cup?', A='white', B='red', C='blue',
                         D='black', answer='B', category='direct_attributes', image_path=f'{i}.jpg'))
    for i in range(4, 6):
        rows.append(dict(index=i, question='Is the dog on the left or right side of the cat?', A='left',
                         B='right', C=None, D=None, answer='A', category='relative_position',
                         image_path=f'{i}.jpg'))
    preds = [
        'The cup is red. \\boxed{B}',               # rule-based matcher can read this
        REASONING + 'The cup is red. \\boxed{B}',   # stray "A" makes the rules give up
        REASONING + 'The cup is white. \\boxed{A}', # wrong
        REASONING + 'It looks red to me.',           # no boxed answer
        '\\boxed{A}',
        REASONING + 'The dog is on the left. \\boxed{A}',
    ]
    meta = pd.DataFrame(rows)
    pred = meta.drop(columns=['image_path']).assign(prediction=preds)
    return meta, pred


class StubJudge:
    """Stands in for the judge LLM: answers with the \\boxed{} letter found in the prompt."""

    def working(self):
        return True

    def generate(self, prompt):
        choice = sv.last_boxed(prompt.split('Answer: ')[-1])
        return choice or 'Z'


def evaluate(tmp_path, monkeypatch, judge):
    meta, pred = synthetic_vstar()
    dataset = object.__new__(ImageMCQDataset)
    dataset.dataset_name, dataset.data, dataset.meta_only = 'VStarBench', meta, True

    run_dir = tmp_path / 'CapImagine-7B' / 'T20260301_G0bac5c06'
    run_dir.mkdir(parents=True)
    eval_file = str(run_dir / 'CapImagine-7B_VStarBench.xlsx')
    dump(pred, eval_file)
    monkeypatch.setattr('vlmeval.dataset.image_mcq.build_judge', lambda **kw: StubJudge())
    dataset.evaluate(eval_file, model=judge, nproc=1)
    return sv.main(['--work-dir', str(tmp_path), '--judge', judge])


def test_vstar_prompt_matches_protocol():
    meta, _ = synthetic_vstar()
    dataset = object.__new__(ImageMCQDataset)
    dataset.meta_only = True
    msgs = dataset.build_prompt(meta.iloc[4])
    assert msgs[0] == dict(type='image', value='4.jpg')
    assert msgs[1]['value'] == ('Question: Is the dog on the left or right side of the cat?\n'
                                'Options:\nA. left\nB. right\n'
                                'Please select the correct answer from the options above. \n')


def test_judge_protocol(tmp_path, monkeypatch):
    summary = evaluate(tmp_path, monkeypatch, 'chatgpt-0125')
    assert summary['splits']['Attribute']['correct'] == 2
    assert summary['splits']['Spatial']['correct'] == 2
    assert summary['n_exact_match_fallback'] == 0


def test_exact_matching_misses_reasoning_answers(tmp_path, monkeypatch):
    summary = evaluate(tmp_path, monkeypatch, 'exact_matching')
    # Only the short outputs survive the rule-based matcher.
    assert summary['splits']['Attribute']['correct'] == 1
    assert summary['splits']['Spatial']['correct'] == 1
    assert summary['n_exact_match_fallback'] >= 2
