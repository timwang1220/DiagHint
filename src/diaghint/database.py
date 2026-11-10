"""Read-only PostgreSQL inspection and provenance-scoped plan timing cache."""
from pathlib import Path
import time
import psycopg2
from .storage import digest,save,read
from .plans import query,aliases,static,root


class Database:
    def __init__(self,config,output):
        cfg=config['database']
        if not cfg.get('revision') or cfg['revision']=='to be filled':raise ValueError('Fill in database.revision in the configuration file')
        self.output=Path(output)
        self.connection=psycopg2.connect(host=cfg['host'],port=cfg['port'],dbname=cfg['name'],user=cfg['user'],password=cfg['password'],application_name='diaghint')
        self.connection.autocommit=True
        with self.connection.cursor() as c:
            c.execute('SET default_transaction_read_only=on');c.execute("LOAD 'pg_hint_plan'")
            c.execute('SELECT current_database(),current_user,version()');identity=c.fetchone()
            c.execute("SELECT name,setting FROM pg_settings WHERE name NOT IN ('application_name','statement_timeout') ORDER BY name");settings=c.fetchall()
            c.execute("SELECT c.oid,c.relname,c.relfilenode,c.relpages,c.reltuples,pg_get_indexdef(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname NOT IN ('pg_catalog','information_schema') ORDER BY c.oid")
            catalog=c.fetchall()
        self.scope=digest(dict(identity=identity,settings=settings,catalog=catalog,revision=cfg['revision']))
        saved=self.output/'database_scope.json'
        if saved.exists() and read(saved)['scope']!=self.scope:
            self.connection.close();raise RuntimeError('Database/settings changed: use a new artifact directory')
        save(saved,dict(scope=self.scope,revision=cfg['revision']))

    def close(self):self.connection.close()

    def metadata(self,sql):
        result={}
        with self.connection.cursor() as c:
            for alias,relation in aliases(sql).items():
                c.execute('SELECT to_regclass(%s)::oid',(relation,));oid=c.fetchone()[0]
                if oid is None:raise ValueError('A referenced table is unavailable')
                c.execute('SELECT attname FROM pg_attribute WHERE attrelid=%s AND attnum>0 AND NOT attisdropped ORDER BY attnum',(oid,));columns=[r[0] for r in c.fetchall()]
                c.execute('SELECT ci.relname,pg_get_indexdef(i.indexrelid),i.indisunique,pg_get_expr(i.indpred,i.indrelid),pg_get_expr(i.indexprs,i.indrelid),ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY k(attnum,pos) JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum ORDER BY k.pos) FROM pg_index i JOIN pg_class ci ON ci.oid=i.indexrelid WHERE i.indrelid=%s AND i.indisvalid AND i.indisready ORDER BY ci.relname',(oid,))
                indexes=[dict(name=r[0],definition=r[1],unique=r[2],predicate=r[3],expression=r[4],columns=r[5]) for r in c.fetchall()]
                result[alias]=dict(relation=relation,columns=columns,indexes=[r['name'] for r in indexes],index_definitions=indexes)
        return result

    def explain(self,sql,hint='',analyze=False,timeout_ms=300000):
        query(sql)
        from .hints import comment
        comment(hint)
        with self.connection.cursor() as c:
            c.execute('SET statement_timeout='+str(int(timeout_ms)))
            self.connection.notices.clear();started=time.perf_counter()
            try:
                options='ANALYZE, BUFFERS, SETTINGS, VERBOSE, TIMING OFF, SUMMARY ON, FORMAT JSON' if analyze else 'SETTINGS, VERBOSE, FORMAT JSON'
                c.execute('EXPLAIN ('+options+') '+hint+' '+sql)
                doc=c.fetchone()[0][0]
                return dict(status='ok',document=doc,execution_ms=doc.get('Execution Time'),planning_ms=doc.get('Planning Time'),
                            wall_ms=1000*(time.perf_counter()-started),notices=list(self.connection.notices))
            except psycopg2.Error as error:
                return dict(status='timeout' if error.pgcode=='57014' else 'sql_error',execution_ms=None,planning_ms=None,
                            wall_ms=1000*(time.perf_counter()-started),sqlstate=error.pgcode,censored=error.pgcode=='57014',timeout_ms=timeout_ms)

    def measure(self,sql,hint,folder,timeout_ms=300000):
        folder=Path(folder)
        if (folder/'result.json').exists():return read(folder/'result.json')
        probe=self.explain(sql,hint,False,timeout_ms);save(folder/'explain.json',probe)
        if probe['status']!='ok':save(folder/'result.json',probe);return probe
        key=digest(dict(sql=sql,scope=self.scope,plan=static(probe['document'])))
        cache_path=self.output/'plan_cache'/f'{key}.json'
        if cache_path.exists():
            result=dict(read(cache_path),cached=True,time_source=str(cache_path),probe_wall_ms=probe['wall_ms'])
        else:
            measured=self.explain(sql,hint,True,timeout_ms);save(folder/'analyze.json',measured)
            if measured['status']=='ok' and static(measured['document'])!=static(probe['document']):
                result=dict(measured,status='plan_changed_between_probe_and_execution',execution_ms=None,cached=False)
            else:
                result=dict(measured,key=key,cached=False,time_source=str(folder/'analyze.json'),probe_wall_ms=probe['wall_ms'])
                if measured['status']=='ok':save(cache_path,result)
        save(folder/'result.json',result);return result
