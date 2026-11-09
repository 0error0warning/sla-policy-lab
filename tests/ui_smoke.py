"""Optional browser smoke test. Uses existing Playwright + Chromium; no installs."""
import json, os, subprocess, time
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent.parent
PORT=8878
server=subprocess.Popen(['python','run.py','--port',str(PORT)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
try:
    for _ in range(100):
        try:
            with urlopen(f'http://127.0.0.1:{PORT}/api/health',timeout=1):break
        except OSError:time.sleep(.1)
    else:raise RuntimeError('Local server did not become ready')
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path=os.environ.get('CHROMIUM_PATH','/usr/bin/chromium'),args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(f'http://127.0.0.1:{PORT}/')
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('完成。')")
        assert page.locator('.method-card').count()==4
        assert page.locator('#metrics tr').count()==5
        assert page.locator('#chart svg path').count()==5
        assert page.locator('body').evaluate('(el)=>el.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(ROOT/'artifacts/desktop.png'),full_page=True)
        page.locator('#profile').select_option('preserve');page.locator('#partner').select_option('supportive')
        page.locator('#strength').evaluate("el => { el.value = '0'; el.dispatchEvent(new Event('input', {bubbles:true})); }");page.locator('#count').fill('4')
        page.locator('#run').click();page.wait_for_function("document.querySelector('#status').textContent.startsWith('完成。4')")
        assert 'preserve' in page.locator('#run-meta').inner_text()
        page.locator('#turn').evaluate("el => { el.value = '10'; el.dispatchEvent(new Event('input', {bubbles:true})); }");assert page.locator('#turn-output').inner_text()=='回合 11'
        page.locator('#run').click();page.wait_for_function("document.querySelector('#status').textContent.startsWith('完成。4')")
        assert page.locator('#turn').input_value()=='0'
        with page.expect_download() as dl:page.locator('#export').click()
        path=ROOT/'artifacts/ui-export.json';dl.value.save_as(str(path));data=json.loads(path.read_text())
        assert data['config']['profile']=='preserve' and data['steering_strength']==0
        assert data['methods']['steering']['trials'][0]['trajectory']==data['methods']['frozen_base']['trials'][0]['trajectory']
        page.route('**/api/run',lambda route:route.abort());page.locator('#run').click()
        page.wait_for_function("document.querySelector('#status').classList.contains('error')")
        assert page.locator('#run').is_enabled() and page.locator('#export').is_enabled()
        page.unroute('**/api/run');page.locator('#run').click()
        page.wait_for_function("document.querySelector('#status').textContent.startsWith('完成。4')")
        page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(ROOT/'artifacts/mobile.png'),full_page=True)
        assert page.locator('body').evaluate('(el)=>el.scrollWidth <= window.innerWidth')
        assert not errors,errors
        browser.close()
        print('PASS: desktop/mobile layout; four mechanisms + frozen base; replay; changed/repeated runs; export; alpha=0 ablation; failure recovery; zero JS exceptions.')
finally:
    server.terminate();server.wait(timeout=10)
