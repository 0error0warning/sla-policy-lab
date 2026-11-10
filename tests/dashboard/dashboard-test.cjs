'use strict';
// Isolated DOM/event simulation, intentionally not a browser or layout engine.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const [repo, generated] = process.argv.slice(2);
if (!repo || !generated) throw Error('Run through run_checks.py or pass REPO GENERATED_DIRECTORY');
const load = name => JSON.parse(fs.readFileSync(path.join(generated, name+'.json'),'utf8'));
const baseline = load('baseline'), support = load('support'), single = load('single'), online = load('online'), onlineOff = load('online_off');
let document, downloadCount = 0;
class TextNode {
  constructor(text) { this.text = String(text); this.parentNode = null; }
  get textContent() { return this.text; }
  set textContent(value) { this.text = String(value); }
  remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(node => node !== this); }
}
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.attributes = {}; this.children = []; this.listeners = {};
    this._value = ''; this.disabled = false; this.dataset = {}; this.classes = new Set();
    this.style = {setProperty:(key,value) => { this.style[key] = value; }};
    this.classList = {
      add:(...keys) => keys.forEach(key => this.classes.add(key)),
      remove:(...keys) => keys.forEach(key => this.classes.delete(key)),
      contains:key => this.classes.has(key),
      toggle:(key,value) => { const on = value === undefined ? !this.classes.has(key) : value; if (on) this.classes.add(key); else this.classes.delete(key); return on; }
    };
  }
  set className(value) { this.classes = new Set(value.split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classes].join(' '); }
  set value(value) { this._value = String(value); }
  get value() { return this._value; }
  get firstChild() { return this.children[0]; }
  append(...nodes) {
    for (let node of nodes) {
      if (typeof node === 'string') node = new TextNode(node);
      node.parentNode = this; this.children.push(node);
      if (this.tagName === 'SELECT' && node.tagName === 'OPTION' && (this.children.length === 1 || node.attributes.selected !== undefined)) this.value = node.value;
    }
  }
  replaceChildren(...nodes) { for (const node of this.children) node.parentNode = null; this.children = []; this.append(...nodes); }
  get textContent() { return this.children.map(node => node.textContent).join(''); }
  set textContent(value) { this.replaceChildren(new TextNode(value)); }
  setAttribute(key,value) {
    this.attributes[key] = String(value);
    if (key === 'id') this.id = String(value);
    if (key === 'class') this.className = String(value);
    if (key === 'value') this.value = value;
    if (['min','max','type'].includes(key)) this[key] = String(value);
    if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g,(_,letter) => letter.toUpperCase())] = String(value);
  }
  getAttribute(key) { return this.attributes[key] ?? null; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(key,callback) { (this.listeners[key] ??= []).push(callback); }
  async trigger(key) { for (const callback of this.listeners[key] ?? []) await callback({preventDefault(){},target:this}); }
  click() { if (this.tagName === 'A') downloadCount++; return this.trigger('click'); }
  requestSubmit() { return this.trigger('submit'); }
  querySelectorAll(selector) { return walk(this).filter(node => node !== this && matches(node,selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  focus() { document.activeElement = this; }
  remove() { TextNode.prototype.remove.call(this); }
}
function walk(root) { return [root,...root.children.flatMap(node => node instanceof Element ? walk(node) : [])]; }
function matches(node,selector) {
  if (selector.startsWith('#')) return node.id === selector.slice(1);
  if (selector.startsWith('.')) return node.classes.has(selector.slice(1));
  if (selector.startsWith('[')) return node.attributes[selector.slice(1,-1)] !== undefined;
  return node.tagName === selector.toUpperCase();
}
function build(node) {
  if ('text' in node) return new TextNode(node.text);
  const element = new Element(node.tag);
  for (const [key,value] of Object.entries(node.attrs)) element.setAttribute(key,value ?? '');
  for (const child of node.children) element.append(build(child));
  return element;
}
const dom = build(load('dom')), elements = walk(dom);
const ids = Object.fromEntries(elements.filter(node => node.id).map(node => [node.id,node]));
document = {
  body:elements.find(node => node.tagName === 'BODY'), activeElement:null, hidden:false, events:{},
  getElementById:id => ids[id], createElement:tag => new Element(tag),
  createElementNS:(_,tag) => new Element(tag), createTextNode:text => new TextNode(text),
  querySelectorAll:selector => walk(dom).filter(node => matches(node,selector)),
  addEventListener(key,callback) { this.events[key] = callback; }
};
// Keep the request and real backend fixture sizes consistent.
ids.count.value = '3';
let timerID = 0, intervals = new Map(), timeouts = new Map(), failFetch = false, malformed = false, payloads = [];
const media = {matches:false, addEventListener(key,callback) { this.listener = callback; }};
function fixtureFor(payload) {
  const candidate = payload.mode === 'online' ? (payload.learning_enabled ? online : onlineOff) : payload.seed_count === 1 ? single : payload.config.profile === 'preserve' ? support : baseline;
  assert.equal(payload.mode,candidate.mode,'fixture mode must match request');
  assert.equal(payload.online_lr,candidate.learning.learning_rate,'fixture learning rate must match request');
  assert.equal(payload.learning_enabled,candidate.learning.enabled,'fixture learning state must match request');
  assert.equal(payload.seed_start,candidate.seed_start,'fixture seed must match request');
  assert.equal(payload.seed_count,candidate.seed_count,'fixture count must match request');
  assert.equal(payload.strength,candidate.steering_strength,'fixture strength must match request');
  for (const [key,value] of Object.entries(candidate.config)) assert.equal(payload.config[key],value,'fixture config '+key+' must match request');
  return candidate;
}
const context = {
  document, window:{matchMedia:() => media,addEventListener(){}}, console, Blob, AbortController,
  URL:{createObjectURL:() => 'blob:testing',revokeObjectURL(){}},
  setTimeout:callback => { timeouts.set(++timerID,callback); return timerID; }, clearTimeout:id => timeouts.delete(id),
  setInterval:(callback,delay) => { intervals.set(++timerID,{callback,delay}); return timerID; }, clearInterval:id => intervals.delete(id),
  fetch:async(url,options) => {
    if (url === '/api/health') return {ok:true,json:async() => ({status:'ready'})};
    const payload = JSON.parse(options.body); payloads.push(payload);
    if (failFetch) throw Error('simulated network failure');
    return {ok:true,json:async() => malformed ? {methods:{frozen_base:{}},trajectory_scope:'all_seeds'} : fixtureFor(payload)};
  }
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(repo,'web/app.js'),'utf8'),context);
const get = id => ids[id], flush = () => new Promise(resolve => setImmediate(resolve));
(async() => {
  await flush(); await flush();
  assert(get('status').textContent.startsWith('完成。3'));
  assert.equal(get('method-cards').children.length,4); assert.equal(get('metrics').children.length,5);
  assert.equal(get('replay').children.length,5); assert.equal(get('replay-seed').children.length,3);
  assert.equal(get('chart').querySelectorAll('path').length,5,'damage chart has five paths; markers are circles');
  assert.equal(intervals.size,0,'must not autoplay'); assert.equal(payloads[0].trace_all,true);
  assert.equal(get('play').disabled,false);
  await get('play').click(); assert.equal(intervals.size,1); assert.equal(get('play').getAttribute('aria-pressed'),'true');
  intervals.values().next().value.callback(); assert.equal(get('turn').value,'1'); assert.equal(get('turn-output').textContent,'回合 2');
  get('speed').value = '300'; await get('speed').trigger('change'); assert.equal(intervals.size,1); assert.equal(intervals.values().next().value.delay,300);
  await get('step-next').click(); assert.equal(intervals.size,0); assert.equal(get('turn').value,'2');
  const timelineRows = get('timeline').children.filter(node => node.classList.contains('timeline-row'));
  assert.equal(timelineRows.length,5);
  await timelineRows[2].children[1].children[7].click(); assert.equal(get('turn').value,'7'); assert.equal(get('focus-method').textContent,'Toy LoRA');
  get('legend').children[3].focus(); await get('legend').children[3].click(); assert.equal(get('focus-method').textContent,'Act. Steering'); assert.equal(document.activeElement,get('legend').children[3]);
  get('replay').children[1].focus(); await get('replay').children[1].click(); assert.equal(document.activeElement,get('replay').children[1]);
  await get('play').click(); get('replay-seed').value = '1'; await get('replay-seed').trigger('change');
  assert.equal(intervals.size,0); assert.equal(get('turn').value,'0'); assert.equal(get('chart-seed').textContent,'SEED 101');
  const shortest = Object.entries(baseline.methods).sort((a,b) => a[1].trials[1].trajectory.length-b[1].trials[1].trajectory.length)[0][0];
  const longest = Math.max(...Object.values(baseline.methods).map(method => method.trials[1].trajectory.length));
  get('turn').value = String(longest-1); await get('turn').trigger('input');
  await get('legend').children[Object.keys(baseline.methods).indexOf(shortest)].click(); assert(get('focus-content').textContent.includes('本条轨迹已结束'));
  await get('restart').click(); assert.equal(get('turn').value,'0'); await get('play').click();
  document.hidden = true; document.events.visibilitychange(); assert.equal(intervals.size,0); document.hidden = false;
  await get('play').click(); get('load').value = '.9'; await get('load').trigger('input');
  assert.equal(intervals.size,0); assert(get('config-state').classList.contains('dirty')); assert(get('status').textContent.includes('参数已更改'));
  await document.querySelectorAll('[data-preset]')[2].click();
  assert.equal(get('profile').value,'preserve'); assert.equal(get('duration').value,'5'); assert.equal(get('duration-output').textContent,'5 回合');
  await get('experiment-form').trigger('submit'); assert(get('status').textContent.startsWith('完成。3'));
  assert.equal(payloads.at(-1).config.protection_duration,5); assert.equal(payloads.at(-1).config.profile,'preserve');
  assert.equal(get('turn').value,'0'); assert.equal(get('replay-seed').value,'0'); assert(!get('config-state').classList.contains('dirty'));
  assert.equal(get('method-cards').children[1].children[2].firstChild.textContent,support.methods.small_policy.summary.lifespan.mean.toFixed(1));
  failFetch = true; await get('experiment-form').trigger('submit');
  assert(get('status').classList.contains('error')); assert.equal(get('run').disabled,false); assert.equal(get('export').disabled,false); assert.equal(get('method-cards').children.length,4);
  failFetch = false; await get('experiment-form').trigger('submit'); assert(!get('status').classList.contains('error'));
  malformed = true; await get('experiment-form').trigger('submit');
  assert(get('status').classList.contains('error')); assert.equal(get('method-cards').children.length,4); assert.equal(get('export').disabled,false);
  malformed = false; await get('experiment-form').trigger('submit');
  await get('export').click(); assert.equal(downloadCount,1);
  media.matches = true; media.listener(); assert(get('motion-note').textContent.includes('减少动态')); assert.equal(intervals.size,0);
  get('turn').value = String(get('turn').max-1); await get('turn').trigger('input'); await get('play').click();
  intervals.values().next().value.callback(); assert.equal(intervals.size,0); assert.equal(get('play').textContent,'↺ 重播');
  await get('play').click(); assert.equal(get('turn').value,'0'); assert.equal(intervals.size,1); await get('play').click();
  for (const [key,value] of Object.entries(single.config)) get(({initial_crack:'crack',protection_duration:'duration'})[key] || key).value = value;
  get('count').value = single.seed_count; get('strength').value = single.steering_strength;
  await get('experiment-form').trigger('submit'); assert(get('status').textContent.startsWith('完成。1'));
  assert(get('delta-chart').textContent.includes('单个 seed')); assert.equal(get('replay-seed').children.length,1);
  assert.deepEqual(single.methods.steering.trials[0].trajectory,single.methods.frozen_base.trials[0].trajectory,'actual alpha-zero trajectory equals frozen base');
  for(const [key,value] of Object.entries(online.config)) get(({initial_crack:'crack',protection_duration:'duration'})[key] || key).value = value;
  get('count').value=online.seed_count; get('strength').value=online.steering_strength;
  get('mode').value='online'; await get('mode').trigger('change');
  get('online-lr').value='.02'; get('learning-enabled').value='true';
  await get('experiment-form').trigger('submit'); assert(get('status').textContent.startsWith('完成。2'));
  await get('replay').children[1].click();
  assert(get('training-summary').textContent.includes('Online'));
  assert(get('focus-content').querySelector('.online-evidence'),'online replay must display update evidence');
  assert(get('method-cards').children[1].textContent.includes('平均更新'));
  assert.equal(online.methods.small_policy.trials[0].learning_updates,12);
  assert(online.methods.small_policy.trials[0].parameter_delta_l2>0);
  await get('step-next').click(); assert.equal(get('turn').value,'1');
  get('learning-enabled').value='false'; await get('learning-enabled').trigger('change');
  await get('experiment-form').trigger('submit'); assert(get('status').textContent.startsWith('完成。2'));
  assert(get('training-summary').textContent.includes('学习关闭'));
  assert.equal(onlineOff.methods.small_policy.trials[0].learning_updates,0);
  assert.equal(onlineOff.methods.lora.trials[0].parameter_delta_l2,0);
  const behavior=trial=>trial.trajectory.map(({learning,...step})=>step);
  assert.deepEqual(behavior(onlineOff.methods.small_policy.trials[0]),behavior(onlineOff.methods.frozen_base.trials[0]));
  assert.deepEqual(behavior(onlineOff.methods.lora.trials[0]),behavior(onlineOff.methods.frozen_base.trials[0]));
  assert.equal(timeouts.size,1,'only deferred blob URL cleanup remains');
  console.log('PASS: actual-result rendering; five curves; 4 method cards; play/pause/step/restart/speed; seed reset; terminal state; focus retention; timeline selection; visibility pause; dirty state; presets; matching API payload; repeated run; network failure; malformed response; export; reduced motion; playback end; single-seed CI; real alpha-zero equivalence; online mode switch/evidence; learning-disabled frozen controls.');
  console.log('Scope: isolated DOM/event harness, not browser layout or screenshot verification.');
})().catch(error => { console.error(error); process.exitCode = 1; });
