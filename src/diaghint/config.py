import configparser
from pathlib import Path
from .storage import digest


def load(path):
    path=Path(path).resolve()
    parser=configparser.ConfigParser(interpolation=None)
    with path.open() as stream:parser.read_file(stream)
    config={
        'database':dict(parser['database']),
        'llm':dict(parser['llm']),
        'pipeline':dict(parser['pipeline']),
    }
    config['database']['port']=parser.getint('database','port')
    config['llm']['temperature']=parser.getfloat('llm','temperature')
    config['llm']['request_timeout_seconds']=parser.getint('llm','request_timeout_seconds')
    config['llm']['max_requests']=parser.getint('llm','max_requests')
    config['pipeline']['seed']=parser.getint('pipeline','seed')
    if config['llm'].get('temperature')!=0:raise ValueError('This pipeline fixes temperature=0')
    return config


def public_snapshot(config):
    return {section:{k:v for k,v in values.items() if k not in {'api_key','password'}} for section,values in config.items()}


def protocol_id(config):
    return digest(public_snapshot(config))
