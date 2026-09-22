"""Fail fast when the VLMEvalKit judge LLM is unreachable.

At the pinned VLMEvalKit commit, an MCQ evaluation whose judge API does not
respond silently falls back to exact matching, yet still names its output
after the judge (e.g. ``*_openai_result.xlsx``). CapImagine answers with long
reasoning chains that the rule-based matcher often cannot parse, so that
fallback would quietly lower the V* score. Run this before evaluation.
"""
import argparse
import sys

from vlmeval.dataset.utils import build_judge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge', default='chatgpt-0125', help='VLMEvalKit judge name (default: %(default)s)')
    args = parser.parse_args()

    judge = build_judge(model=args.judge, retry=3)
    if not judge.working():
        sys.exit(
            f'Judge "{args.judge}" ({judge.model}) is not responding.\n'
            'Put OPENAI_API_KEY=... (and, for a non-OpenAI endpoint, OPENAI_API_BASE=<.../v1/chat/completions>\n'
            'plus LOCAL_LLM=<served model name>) in VLMEvalKit/.env, or pick another judge with JUDGE=<name>.'
        )
    print(f'Judge OK: {args.judge} -> {judge.model}')


if __name__ == '__main__':
    main()
