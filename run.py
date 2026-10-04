"""Run one model × condition over the 1,500 CES 2021 profiles (Together or a mock backend)."""
import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
import pandas as pd
from build import ROOT, QUESTIONS, COLUMNS, GROUPS
from build21 import CONDITIONS
from survey import load_questions, load_profiles, messages, parse, INVALID

MODELS = {'small': 'Qwen/Qwen3.5-9B', 'large': 'Qwen/Qwen3.8-2.4T-A95B'}
URL = 'https://api.together.xyz/v1/chat/completions'
SETTINGS = {'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 1024, 'chat_template_kwargs': {'enable_thinking': False}}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 8


def mock_reply(synthetic_id, items):
    rng = random.Random(synthetic_id)
    lines = [f'{n}: {rng.choice(list(item["options"]))}' for n, item in enumerate(items, 1)]
    if rng.random() < 0.05:
        lines[rng.randrange(len(lines))] = 'not sure'
    return '\n'.join(lines), {'prompt_tokens': 0, 'completion_tokens': 0}, 'stop'


async def together_reply(client, key, model, chat):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.post(URL, headers={'Authorization': f'Bearer {key}'},
                                         json={'model': model, 'messages': chat, **SETTINGS}, timeout=120)
        except httpx.TransportError as error:
            failure = repr(error)
        else:
            if response.status_code == 200:
                body = response.json()
                choice = body['choices'][0]
                return choice['message'].get('content') or '', body.get('usage', {}), choice.get('finish_reason'), attempt
            if response.status_code not in RETRY_STATUS:
                raise RuntimeError(f'HTTP {response.status_code}: {response.text[:500]}')
            failure = f'HTTP {response.status_code}'
        # Only failed requests are retried, never undesired answers.
        await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f'Gave up after {MAX_ATTEMPTS} attempts: {failure}')


async def run(args, todo, items, raw_path):
    key = None
    if args.backend == 'together':
        key = os.environ.get('TOGETHER_API_KEY') or (args.key_file and Path(args.key_file).expanduser().read_text().strip())
        if not key:
            raise SystemExit('Set TOGETHER_API_KEY or pass --key-file')
    semaphore = asyncio.Semaphore(args.concurrency)
    done = 0
    async with httpx.AsyncClient() as client:
        async def one(profile):
            nonlocal done
            chat = messages(profile, args.condition, items)
            async with semaphore:
                started = time.monotonic()
                if args.backend == 'mock':
                    text, usage, finish = mock_reply(profile['synthetic_id'], items)
                    attempts = 1
                else:
                    text, usage, finish, attempts = await together_reply(client, key, MODELS[args.model], chat)
            record = {'synthetic_id': profile['synthetic_id'], 'text': text, 'usage': usage, 'finish_reason': finish,
                      'attempts': attempts, 'seconds': round(time.monotonic() - started, 2),
                      'at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
            with raw_path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f'  {done}/{len(todo)} calls', flush=True)
        await asyncio.gather(*(one(profile) for profile in todo.to_dict('records')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--backend', choices=['together', 'mock'], default='together')
    parser.add_argument('--per-group', type=int, default=100, help='profiles per group; below 100 is a smoke test')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--key-file', help='file holding the Together API key (else TOGETHER_API_KEY)')
    args = parser.parse_args()

    items = load_questions()
    profiles = load_profiles()
    profiles = profiles.groupby(['Region', 'Age'], sort=False).head(args.per_group)
    name = f'{"mock_" if args.backend == "mock" else ""}{args.model}_{args.condition}'
    if args.per_group < 100:
        name += f'_smoke{args.per_group}'
    raw_path = ROOT / 'raw' / f'{name}.jsonl'
    raw_path.parent.mkdir(exist_ok=True)
    seen = set()
    if raw_path.exists():
        seen = {json.loads(line)['synthetic_id'] for line in raw_path.read_text().splitlines() if line.strip()}
    todo = profiles[~profiles.synthetic_id.isin(seen)]
    print(f'{name}: {len(profiles)} profiles, {len(seen)} already done, {len(todo)} to call')
    asyncio.run(run(args, todo, items, raw_path))

    records = {}
    for line in raw_path.read_text().splitlines():
        record = json.loads(line)
        records.setdefault(record['synthetic_id'], record)  # keep the first reply per profile
    answers = pd.DataFrame([{'cps25_ResponseId': sid, **parse(records[sid]['text'], items)} for sid in profiles.synthetic_id])
    result = answers.assign(Age=profiles.Age.to_numpy(), Region=profiles.Region.to_numpy())[COLUMNS]
    invalid = (result[QUESTIONS] == INVALID).mean()
    tokens = {kind: sum(r['usage'].get(kind) or 0 for r in records.values()) for kind in ('prompt_tokens', 'completion_tokens')}
    if args.per_group == 100:
        assert len(result) == 1500 and set(zip(result.Region, result.Age)) == set(GROUPS)
        result.to_csv(ROOT / f'results_{name}.csv', index=False)

    log_path = ROOT / 'runs.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log[name] = {
        'model': MODELS[args.model] if args.backend == 'together' else 'mock', 'provider': args.backend, 'condition': args.condition,
        'fields': CONDITIONS[args.condition], 'profiles': len(profiles), 'settings': SETTINGS,
        'prompt_sha256': hashlib.sha256((ROOT / 'prompt.txt').read_bytes()).hexdigest(),
        'first_call': min(r['at'] for r in records.values()), 'last_call': max(r['at'] for r in records.values()),
        'tokens': tokens, 'invalid_rate': round(float(invalid.mean()), 4), 'invalid_by_item': invalid.round(4).to_dict(),
        'finish_reasons': pd.Series([r['finish_reason'] for r in records.values()]).value_counts().to_dict(),
        'retried_calls': sum(r['attempts'] > 1 for r in records.values()),
    }
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + '\n')
    print(f'Invalid rate {invalid.mean():.1%}; tokens {tokens}; logged in runs.json')


if __name__ == '__main__':
    main()
