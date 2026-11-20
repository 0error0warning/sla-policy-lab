from pathlib import Path, PurePosixPath
from hashlib import sha256
import json, os, subprocess, sys, time, traceback, zipfile

slug = 'sla-ib-frozeninterface'
root = Path('/content') / (slug + '-v1')
archive = Path('/content') / (slug + '-inputs.zip')
assert sha256(archive.read_bytes()).hexdigest() == 'ea7ff5bce441a7bf7d157effecccc150f3e1da46389b650d38c38c665787a1fb'
root.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n for n in z.namelist())
    z.extractall(root)
lock=json.loads((root/'run-lock.json').read_bytes())
assert all(sha256((root/n).read_bytes()).hexdigest()==v for n,v in lock['files'].items())
env=dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
receipt=dict(stage='dependencies',input_sha256='ea7ff5bce441a7bf7d157effecccc150f3e1da46389b650d38c38c665787a1fb',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
def status(stage):
    receipt['stage']=stage
    (root/'launch-receipt.json').write_text(json.dumps(receipt))
status('dependencies')
allocated_start=time.time()
code=1
try:
    with (root/'install.log').open('w') as log:
        for args in (['uninstall','-y','torchao'],['install','--disable-pip-version-check','-r','requirements-gpu.txt']):
            subprocess.run([sys.executable,'-m','pip',*args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=480,check=True)
    status('worker')
    env['SLA_WORKER_DEADLINE']=str(allocated_start+8280)
    with (root/'worker.log').open('w') as log:
        code=subprocess.run([sys.executable,'-m','experiments.frozen_interface.gpu','--mode','frozeninterface','--root',str(root)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,allocated_start+8310-time.time())).returncode
except BaseException:
    (root/'launch-failure.txt').write_text(traceback.format_exc())
finally:
    status('export')
    paths=[p for p in (root/'output').rglob('*') if p.is_file()]+[p for p in root.glob('*') if p.name in ('launch-receipt.json','install.log','worker.log','launch-failure.txt','run-lock.json')]
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest()) for p in paths}
    with zipfile.ZipFile(root/'results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(root).as_posix())
        z.writestr('archive-manifest.json',json.dumps(manifest,sort_keys=True))
    result=dict(worker_returncode=code,files=len(paths),bytes=(root/'results.zip').stat().st_size,sha256=sha256((root/'results.zip').read_bytes()).hexdigest(),study_stage='frozeninterface')
    parts=[]
    with (root/'results.zip').open('rb') as f:
        while payload:=f.read(5*1024*1024):
            name='part-%03d.bin'%len(parts)
            (root/name).write_bytes(payload)
            parts.append(dict(name=name,bytes=len(payload),sha256=sha256(payload).hexdigest()))
    result['chunks']=parts
    (root/'export-receipt.json').write_text(json.dumps(result))
    print(json.dumps(result),flush=True)
if code:raise RuntimeError('Worker failed; evidence preserved')
