#!/usr/bin/env python3
"""Dependency-free DOM/event checks against real simulator exports. Not browser QA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from html.parser import HTMLParser

class DOMParser(HTMLParser):
    VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    def __init__(self):
        super().__init__()
        self.root = {'tag':'root','attrs':{},'children':[]}
        self.stack = [self.root]
        self.ids = []
    def handle_starttag(self, tag, attrs):
        element = {'tag':tag, 'attrs':dict(attrs), 'children':[]}
        self.stack[-1]['children'].append(element)
        if 'id' in element['attrs']:
            self.ids.append(element['attrs']['id'])
        if tag not in self.VOID:
            self.stack.append(element)
    def handle_endtag(self, tag):
        for index in range(len(self.stack)-1,0,-1):
            if self.stack[index]['tag'] == tag:
                self.stack = self.stack[:index]
                break
    def handle_data(self, data):
        self.stack[-1]['children'].append({'text':data})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = Path(__file__).resolve().parent
    generated = root / 'generated'
    generated.mkdir(exist_ok=True)
    html = (repo/'web/index.html').read_text()
    dom = DOMParser()
    dom.feed(html)
    assert len(dom.ids) == len(set(dom.ids)), 'HTML IDs must be unique'
    import re
    used_ids = set(re.findall(r"\$\('([^']+)'\)", (repo/'web/app.js').read_text()))
    assert used_ids <= set(dom.ids), used_ids-set(dom.ids)
    (generated/'dom.json').write_text(json.dumps(dom.root,ensure_ascii=False))
    print('PASS: unique HTML IDs and all literal JS ID references exist.',flush=True)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0,str(repo))
    from sla_demo.models import ModelSuite
    from sla_demo.environment import Config
    from sla_demo.experiment import run
    suite = ModelSuite(17)
    variants = {
        'baseline': run(suite,Config(),100,3,1.0,True),
        'support': run(suite,Config(horizon=80,load=.75,initial_crack=.12,partner='supportive',profile='preserve',protection_duration=5),100,3,1.0,True),
        'single': run(suite,Config(profile='preserve'),100,1,0.0,True),
        'online': run(suite,Config(horizon=12),100,2,0.0,True,mode='online'),
        'online_off': run(suite,Config(horizon=12),100,2,0.0,True,mode='online',learning_enabled=False),
    }
    for name,data in variants.items():
        (generated/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False))
    print('Generated five actual simulator exports with model '+suite.fingerprint,flush=True)
    subprocess.run(['node','--check',str(repo/'web/app.js')],check=True)
    subprocess.run(['node',str(root/'dashboard-test.cjs'),str(repo),str(generated)],check=True)
    print('Not tested: browser rendering, layout, screenshots, real browser event behavior, or HTTP transport.',flush=True)

if __name__ == '__main__':
    main()
