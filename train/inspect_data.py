"""Check CapImagine-Data before training: schema, image paths, and what the model will see.

Needs no GPU or model. Run it once after downloading; it fails with the offending record if
the layout is not one `train/data.py` understands.

    python -m train.inspect_data --data_path data/CapImagine-Data/<file>.json --image_root data/CapImagine-Data
"""
import argparse
import collections
import json
import os
import re

from train.data import MONET_SYSTEM_PROMPT, CapImagineDataset, load_records

TAGS = ['<think_image>', '</think_image>', '<abs_vis_token>', '<observation>', '\\boxed{']


def _shorten(value, n=300):
    if isinstance(value, str):
        return value if len(value) <= n else value[:n] + f'... [{len(value)} chars]'
    if isinstance(value, list):
        return [_shorten(v, n) for v in value]
    if isinstance(value, dict):
        return {k: _shorten(v, n) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', nargs='+', required=True)
    parser.add_argument('--image_root', help='default: dir of the first data file')
    parser.add_argument('--assistant_images', default='error', choices=['error', 'drop'])
    args = parser.parse_args()
    image_root = args.image_root or os.path.dirname(os.path.abspath(args.data_path[0]))

    raw = load_records(args.data_path)
    print(f'{len(raw)} records. Raw record 0:')
    print(json.dumps(_shorten(raw[0]), indent=2, ensure_ascii=False))

    dataset = CapImagineDataset(args.data_path, image_root, MONET_SYSTEM_PROMPT, args.assistant_images)
    n_images = collections.Counter()
    missing, tag_counts, assistant_chars = [], collections.Counter(), []
    for messages in dataset.samples:
        paths = [i['image'] for m in messages for i in m['content'] if i['type'] == 'image']
        n_images[len(paths)] += 1
        missing += [p for p in paths if not os.path.exists(p)]
        reply = ''.join(i['text'] for m in messages if m['role'] == 'assistant' for i in m['content'])
        assistant_chars.append(len(reply))
        for tag in TAGS:
            tag_counts[tag] += tag in reply

    assistant_chars.sort()
    print(f'\nNormalised sample 0 (image root: {image_root}):')
    print(json.dumps(_shorten(dataset[0]), indent=2, ensure_ascii=False))
    print(f'\nimages per sample: {dict(sorted(n_images.items()))}')
    print(f'assistant reply length (chars): median {assistant_chars[len(assistant_chars) // 2]}, '
          f'max {assistant_chars[-1]}')
    print('samples whose replies contain: ' + ', '.join(f'{t} {c}' for t, c in tag_counts.items()))
    if missing:
        print(f'\nMISSING {len(missing)} image files, e.g.:\n  ' + '\n  '.join(missing[:5]))
        print('Fix --image_root (and unzip the image archive) before training.')
    else:
        print('\nAll image files found.')
    first = re.sub(r'\s+', ' ', next(i['text'] for m in dataset[0] if m['role'] == 'assistant' for i in m['content']))
    print(f'\nFirst assistant reply: {first[:500]}')


if __name__ == '__main__':
    main()
