"""One OpenAI-compatible client; one-shot calls, atomic budget, exact cache."""
from pathlib import Path
import time
from urllib.parse import urlsplit
from .storage import digest,save,read,locked


class LLM:
    def __init__(self,config,root,prompts,cache_roots=()):
        self.config=config['llm'];self.root=Path(root);self.prompts=prompts
        self.cache_roots=[Path(p) for p in cache_roots]
        self.budget=self.root/'llm_budget.json'
        if not self.budget.exists():save(self.budget,dict(maximum=self.config['max_requests'],requests={}))

    def generate(self,phase,context,conversation=()):
        messages=[*conversation,*self.prompts.render(phase,context)]
        contract=dict(base_url=self.config['base_url'],model=self.config['model'],temperature=0,
                      prompt_version=self.prompts.fingerprint,messages=messages,response_format={'type':'json_object'})
        rid=digest(contract);folder=self.root/'requests'/rid
        for cache in self.cache_roots:
            previous=cache/'requests'/rid
            if (previous/'request.json').exists() and (previous/'result.json').exists() and read(previous/'request.json')==contract:
                result=read(previous/'result.json')
                save(folder/'request.json',contract);save(folder/'result.json',result)
                save(folder/'cache_source.json',dict(source=str(previous)))
                return dict(result,cached=True,request_id=rid,messages=messages)
        with locked(self.root/'llm_budget.lock'):
            budget=read(self.budget)
            if rid in budget['requests']:
                result=read(folder/'result.json') if (folder/'result.json').exists() else dict(status='interrupted',text=None,api_ms=0)
                return dict(result,cached=True,request_id=rid,messages=messages)
            if len(budget['requests'])>=budget['maximum']:raise RuntimeError('LLM request budget exhausted')
            budget['requests'][rid]=dict(time=time.time(),phase=phase)
            save(self.budget,budget);save(folder/'request.json',contract)
        start=time.perf_counter()
        try:
            if not self.config['api_key'] or self.config['api_key']=='to be filled':raise ValueError('Fill in llm.api_key in the configuration file')
            from openai import OpenAI
            extra={'thinking':{'type':'disabled'}} if urlsplit(self.config['base_url']).hostname=='api.deepseek.com' else {}
            with OpenAI(api_key=self.config['api_key'],base_url=self.config['base_url'],timeout=self.config['request_timeout_seconds'],max_retries=0) as client:
                response=client.chat.completions.create(model=self.config['model'],messages=messages,temperature=0,
                    max_tokens=8192,response_format={'type':'json_object'},extra_body=extra)
            text=response.choices[0].message.content or ''
            result=dict(status='ok' if text.strip() else 'empty_response',text=text,returned_model=response.model,
                        api_ms=1000*(time.perf_counter()-start),usage=response.usage.model_dump() if response.usage else None)
        except Exception as error:
            result=dict(status='api_failure',text=None,error_type=type(error).__name__,http_status=getattr(error,'status_code',None),api_ms=1000*(time.perf_counter()-start))
        save(folder/'result.json',result)
        if result.get('http_status') in {401,402,403,429}:raise RuntimeError('API credentials/quota unavailable; no retry')
        return dict(result,cached=False,request_id=rid,messages=messages)
