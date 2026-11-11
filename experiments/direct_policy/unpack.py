"""Verify a published GPU archive and safely extract to a new directory."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path,PurePosixPath
import zipfile


def unpack(archive,receipt,destination,allow_failed=False):
    data=archive.read_bytes();meta=json.loads(receipt.read_bytes())
    assert len(data)==meta['bytes'] and sha256(data).hexdigest()==meta['sha256']
    assert meta['worker_returncode']==0 or allow_failed,'failed run requires --allow-failed'
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        manifest=json.loads(z.read('archive-manifest.json'))
        assert z.testzip() is None and len(manifest)==meta['files']
        assert set(z.namelist())==set(manifest)|{'archive-manifest.json'}
        assert len(z.namelist())==len(set(z.namelist()))
        target=destination.resolve()
        payloads={}
        for name in z.namelist():
            rel=PurePosixPath(name);path=target.joinpath(*rel.parts).resolve()
            assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in name
            assert path.is_relative_to(target)
            payload=z.read(name)
            if name!='archive-manifest.json':
                assert len(payload)==manifest[name]['bytes'] and sha256(payload).hexdigest()==manifest[name]['sha256']
            payloads[path]=payload
        destination.mkdir(parents=True,exist_ok=False)
        for path,payload in payloads.items():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return dict(archive_verified=True,**meta)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--allow-failed',action='store_true');a=p.parse_args()
    print(json.dumps(unpack(a.archive,a.receipt,a.destination,a.allow_failed)))
