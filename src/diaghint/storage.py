"""Atomic artifacts and provenance; contains no database or API access."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


@contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def checkpoint(root,stage,**values):
    save(Path(root)/'status.json',dict(stage=stage,time=time.time(),**values))


def artifact_dir(path):
    result=Path(path).resolve()
    repo=Path(__file__).resolve().parents[2]
    if result==repo or repo in result.parents:
        # Never mix private prompts/results/checkpoints with publishable source.
        raise ValueError('Artifact output must be outside the source checkout')
    result.mkdir(parents=True,exist_ok=True)
    return result
