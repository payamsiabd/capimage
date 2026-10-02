"""Load CapImagine-Data and turn each record into Qwen2.5-VL chat messages.

CapImagine-Data was rewritten from Monet-SFT-125K and trained with the Monet codebase
(paper §5.1), so records are expected in Monet's layout:

    {"data": [{"role": "system" | "user" | "assistant",
               "content": [{"type": "image", "image": "<path relative to the image root>"},
                           {"type": "text", "text": "..."}]},
              ...],
     "metadata": {...}}

Also accepted, in case the release differs: the bare turn list; a record whose turns sit
under "messages" or "conversations" with OpenAI-style ``{"role", "content"}`` turns; and
LLaVA-style ``{"from", "value"}`` turns with ``<image>`` markers and the image path(s)
under "image" / "images". Anything else raises ``DataFormatError``.
"""
import json
import os

from torch.utils.data import Dataset

# Monet's SFT preprocessing replaces every system message with this text (src/task.py).
MONET_SYSTEM_PROMPT = 'You are a helpful assistant.'
# Monet marks where an auxiliary image used to sit in an assistant turn.
LATENT_PLACEHOLDER = '<abs_vis_token></abs_vis_token>'
ROLE_ALIASES = {'human': 'user', 'gpt': 'assistant', 'model': 'assistant', 'bot': 'assistant'}


class DataFormatError(ValueError):
    pass


def load_records(paths):
    """Concatenate the records of one or more .json / .jsonl files."""
    records = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                part = [json.loads(line) for line in f if line.strip()]
            else:
                part = json.load(f)
        if isinstance(part, dict):
            lists = [v for v in part.values() if isinstance(v, list)]
            if len(lists) != 1:
                raise DataFormatError(f'{path}: top-level object has {len(lists)} list fields; expected one record list')
            part = lists[0]
        records.extend(part)
    return records


def _image_items(record):
    images = record.get('images', record.get('image'))
    if images is None:
        return []
    return [images] if isinstance(images, str) else list(images)


def _resolve(path, image_root):
    if path.startswith('file://'):
        path = path[len('file://'):]
    if os.path.isabs(path) or path.startswith(('http://', 'https://', 'data:')):
        return path
    return os.path.join(image_root, path)


def _content_items(content, pending_images):
    """Normalise a turn's content into Qwen-style items; LLaVA `<image>` markers consume `pending_images`."""
    if isinstance(content, str):
        items = []
        parts = content.split('<image>')
        for i, part in enumerate(parts):
            if i > 0:
                if not pending_images:
                    raise DataFormatError('more <image> markers than images in the record')
                items.append({'type': 'image', 'image': pending_images.pop(0)})
            if part.strip():
                items.append({'type': 'text', 'text': part.strip('\n') if i > 0 else part})
        return items
    items = []
    for item in content:
        if isinstance(item, str):
            items.append({'type': 'text', 'text': item})
        elif item.get('type') == 'text' or ('text' in item and 'type' not in item):
            items.append({'type': 'text', 'text': item['text']})
        elif item.get('type') in ('image', 'image_url') or 'image' in item or 'image_url' in item:
            src = item.get('image', item.get('image_url', item.get('path')))
            if isinstance(src, dict):
                src = src.get('url')
            if not isinstance(src, str):
                raise DataFormatError(f'image item without a path: {item}')
            items.append({'type': 'image', 'image': src})
        else:
            raise DataFormatError(f'unsupported content item: {item}')
    return items


def normalize_record(record, image_root, system_prompt=MONET_SYSTEM_PROMPT, assistant_images='error'):
    """Return the record as a list of ``{"role", "content": [items]}`` turns with absolute image paths.

    system_prompt:     replaces (or adds) the system turn, as Monet does; None keeps the record's own.
    assistant_images:  CapImagine verbalises intermediate images, so assistant turns should be text only.
                       'error' raises if one carries an image; 'drop' removes such images (and Monet's
                       latent placeholders) and keeps the text.
    """
    if isinstance(record, list):
        turns = record
    else:
        turns = next((record[k] for k in ('data', 'messages', 'conversations') if k in record), None)
        if turns is None:
            raise DataFormatError(f'no turns found; record keys: {sorted(record)}')
    pending_images = [] if isinstance(record, list) else _image_items(record)

    messages = []
    for turn in turns:
        role = turn.get('role', turn.get('from'))
        role = ROLE_ALIASES.get(role, role)
        if role not in ('system', 'user', 'assistant'):
            raise DataFormatError(f'unknown role {role!r} in turn {turn}')
        content = _content_items(turn.get('content', turn.get('value', '')), pending_images)
        if role == 'assistant':
            if any(item['type'] == 'image' for item in content):
                if assistant_images == 'error':
                    raise DataFormatError(
                        'assistant turn contains an image; CapImagine trains on text-only imagination. '
                        'Pass --assistant_images drop to remove these images and keep the text.'
                    )
                content = [item for item in content if item['type'] == 'text']
            for item in content:
                item['text'] = item['text'].replace(LATENT_PLACEHOLDER, '')
        messages.append({'role': role, 'content': content})

    if pending_images:
        # LLaVA records without <image> markers: images go before the first user text.
        first_user = next((m for m in messages if m['role'] == 'user'), None)
        if first_user is None:
            raise DataFormatError('record has images but no user turn')
        first_user['content'] = [{'type': 'image', 'image': p} for p in pending_images] + first_user['content']

    for message in messages:
        for item in message['content']:
            if item['type'] == 'image':
                item['image'] = _resolve(item['image'], image_root)

    if system_prompt is not None:
        messages = [m for m in messages if m['role'] != 'system']
        messages.insert(0, {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]})

    if not any(m['role'] == 'assistant' and any(i['text'].strip() for i in m['content']) for m in messages):
        raise DataFormatError('record has no assistant text to train on')
    return messages


class CapImagineDataset(Dataset):
    """Normalised chat records; items are message lists consumed by `QwenVLSFTCollator`."""

    def __init__(self, paths, image_root, system_prompt=MONET_SYSTEM_PROMPT, assistant_images='error',
                 max_samples=None):
        records = load_records(paths)
        if max_samples:
            records = records[:max_samples]
        self.samples = []
        for i, record in enumerate(records):
            try:
                self.samples.append(normalize_record(record, image_root, system_prompt, assistant_images))
            except DataFormatError as e:
                raise DataFormatError(f'record {i}: {e}\nrecord: {json.dumps(record)[:1000]}') from e

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
