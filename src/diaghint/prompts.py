"""Render the workflow templates from the repository's prompts directory."""
import json
from pathlib import Path
import re
from .storage import digest


class Prompts:
    def __init__(self):
        self.directory=Path(__file__).resolve().parents[2]/'prompts'
        self.templates={name:(self.directory/(name+'.prompt')).read_text()
                        for name in ['system','global','leading','node','optimize']}
        self.fingerprint=digest(self.templates)

    def render(self,phase,context):
        if phase not in {'global','leading','node','optimize'}:raise ValueError('Unknown prompt phase')
        values=dict(context,context=context)
        def replace(match):
            value=values[match[1]]
            return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
        text=re.sub(r'\{\{(\w+)\}\}',replace,self.templates[phase])
        return [dict(role='system',content=self.templates['system']),dict(role='user',content=text)]
