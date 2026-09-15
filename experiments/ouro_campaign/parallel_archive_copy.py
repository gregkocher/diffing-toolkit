"""Copy an immutable archive in resumable SSH ranges and verify its whole SHA256.

The original rsync partial, range files, and failed assembly files are retained.
The caller must quiesce any writer to the supplied prefix before invoking this.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

READ_RANGE = '''import sys
with open(sys.argv[1], "rb") as source:
 source.seek(int(sys.argv[2])); remaining=int(sys.argv[3])
 while remaining:
  block=source.read(min(1024*1024,remaining))
  if not block: raise RuntimeError("Archive ended before requested range")
  sys.stdout.buffer.write(block); remaining-=len(block)
 sys.stdout.buffer.flush()
'''


def file_sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def copy_archive(*, ssh, remote_path, output, expected_size, expected_sha256,
                 streams=4, prefix=None, timeout=10800):
    output = Path(output)
    if not 1 <= streams <= 8 or expected_size <= 0:
        raise ValueError('Expected a positive size and 1–8 streams')
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Expected a lowercase SHA256 digest')
    if output.exists():
        if output.stat().st_size == expected_size and file_sha256(output) == expected_sha256:
            return {'sha256': expected_sha256, 'bytes': expected_size, 'already_complete': True}
        raise ValueError('Output already exists with different content; preserve it and choose a new output path')
    if prefix is not None:
        prefix = Path(prefix)
        if not prefix.is_file() or prefix.stat().st_size > expected_size:
            raise ValueError('Prefix must be an existing file no larger than the archive')
    started = time.monotonic()
    deadline = started + timeout
    parts_dir = output.parent / (output.name + '.parts')
    parts_dir.mkdir(parents=True, exist_ok=True)
    width = (expected_size + streams - 1) // streams
    ranges = [(start, min(width, expected_size-start)) for start in range(0, expected_size, width)]
    plan = {'ssh': ssh, 'remote_path': remote_path, 'bytes': expected_size,
            'sha256': expected_sha256, 'ranges': [list(r) for r in ranges]}
    plan_path = parts_dir / 'PLAN.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError('Existing parts belong to a different archive or transfer plan')
    else:
        plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    prefix_state = None if prefix is None else (prefix.stat().st_size, prefix.stat().st_mtime_ns)
    seeded_bytes = 0
    parts = []
    for index, (start, count) in enumerate(ranges):
        part = parts_dir / f'part_{index:02d}'
        parts.append(part)
        if part.exists() and part.stat().st_size > count:
            raise ValueError('A saved range exceeds its planned length')
        if not part.exists() and prefix is not None and prefix_state[0] > start:
            seed_count = min(count, prefix_state[0]-start)
            with prefix.open('rb') as source, part.open('xb') as dest:
                source.seek(start)
                remaining = seed_count
                while remaining:
                    if time.monotonic() >= deadline: raise TimeoutError('Archive transfer deadline expired')
                    block = source.read(min(8*1024*1024, remaining))
                    if not block: raise RuntimeError('Prefix ended during seeding')
                    dest.write(block); remaining -= len(block)
            seeded_bytes += seed_count
    if prefix is not None and (prefix.stat().st_size, prefix.stat().st_mtime_ns) != prefix_state:
        raise RuntimeError('Prefix changed while seeding; stop its writer before retrying')

    def download(index):
        part = parts[index]
        start, count = ranges[index]
        present = part.stat().st_size if part.exists() else 0
        if present < count:
            remaining_time = deadline-time.monotonic()
            if remaining_time <= 0: raise TimeoutError('Archive transfer deadline expired')
            command = shlex.join(['python3', '-c', READ_RANGE, remote_path,
                                  str(start+present), str(count-present)])
            with part.open('ab') as dest:
                subprocess.run(ssh+[command], stdout=dest, check=True, timeout=remaining_time)
        if part.stat().st_size != count:
            raise RuntimeError(f'Range {index} has an unexpected length')

    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
        list(pool.map(download, range(len(parts))))
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(prefix=output.name+'.assembling.', dir=output.parent,
                                     delete=False) as dest:
        assembled = Path(dest.name)
        for part in parts:
            with part.open('rb') as source:
                while block := source.read(8*1024*1024):
                    if time.monotonic() >= deadline: raise TimeoutError('Archive transfer deadline expired')
                    digest.update(block); dest.write(block)
    if digest.hexdigest() != expected_sha256 or assembled.stat().st_size != expected_size:
        raise RuntimeError(f'Whole-archive verification failed; retained {assembled}')
    if output.exists():
        raise RuntimeError('Output appeared during transfer; retained verified assembly')
    assembled.rename(output)
    receipt = {'sha256': expected_sha256, 'bytes': expected_size,
               'elapsed_seconds': time.monotonic()-started, 'streams': streams,
               'seeded_prefix_bytes': seeded_bytes, 'parts_directory': str(parts_dir),
               'prefix_preserved': None if prefix is None else str(prefix)}
    (parts_dir/'TRANSFER_COMPLETE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--identity', required=True, type=Path)
    parser.add_argument('--remote-path', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--size', required=True, type=int)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--streams', type=int, default=4)
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--timeout', type=float, default=10800)
    args = parser.parse_args()
    ssh = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=20',
           '-o', 'Compression=no', '-i', str(args.identity), '-p', str(args.port),
           'root@'+args.host]
    print(json.dumps(copy_archive(ssh=ssh, remote_path=args.remote_path,
        output=args.output, expected_size=args.size, expected_sha256=args.sha256,
        streams=args.streams, prefix=args.prefix, timeout=args.timeout), indent=2))


if __name__ == '__main__':
    main()
