'use strict';
// Every displayed result is read from the local simulator. Playback never creates data.
const $ = id => document.getElementById(id);
const colors = {prompt_reference:'#20756b',small_policy:'#3563c9',lora:'#aa6a29',steering:'#8562aa',frozen_base:'#6f7785'};
const names = {prompt_reference:'Prompt / Rule',small_policy:'Small Policy',lora:'Toy LoRA',steering:'Activation Steering',frozen_base:'Frozen Base'};
const shortNames = {...names,steering:'Act. Steering'};
const descriptions = {
  prompt_reference:['手写规则基线','固定指令选择目标偏好，手写规则直接决定动作。没有 LLM 推理。','0 个学习参数','规则判断'],
  small_policy:['从零训练的小策略','12 维状态，经 24 维 tanh 隐藏层输出三类动作。','387 个训练参数','12 → 24 → 3'],
  lora:['冻结基座 + 低秩更新','只训练输出投影的 rank-2 更新；基座权重保持冻结。','54 个训练参数','W + AB / rank'],
  steering:['冻结基座 + 隐藏向量','在隐藏激活上加入动作对比方向；不是语言概念。','0 个训练参数 · 24 维方向','h + α · direction'],
  frozen_base:['未修改的冻结基线','LoRA 与 steering 共享的平衡规则 MLP，用作消融对照。','无本次干预','冻结 12 → 24 → 3']
};
const onlineDescriptions = {
  small_policy:['逐轮在线更新完整小策略','从公共平衡基座复制，实际奖励反馈后更新全部权重；每条生命独立重置。','387 个在线参数','state → action → reward → Δθ'],
  lora:['逐轮在线更新低秩 adapter','从零 adapter 开始，每轮只更新 rank-2 A/B；基座始终冻结。','54 个在线参数','W frozen + online ΔAB'],
  steering:['固定隐藏向量对照','使用相同基座与固定方向；有动作采样，但不做在线参数学习。','0 个在线参数','h + α · fixed direction'],
  frozen_base:['同起点的采样冻结组','与完整小策略和零 adapter 共享初始 logits 及动作随机数；不更新权重。','0 个在线参数','shared base · no updates']
};
function descriptionFor(key){return result&&result.mode==='online'&&onlineDescriptions[key]?onlineDescriptions[key]:descriptions[key];}
const actionNames = {answer:'回答',refuse:'拒绝',request_protection:'请求保护'};
const actionLetters = {answer:'A',refuse:'R',request_protection:'P'};
const presets = {
  balanced:{horizon:60,load:.65,crack:.08,partner:'mixed',profile:'balanced',duration:3,strength:1},
  pressure:{horizon:60,load:.9,crack:.2,partner:'unavailable',profile:'balanced',duration:3,strength:1},
  support:{horizon:80,load:.75,crack:.12,partner:'supportive',profile:'preserve',duration:5,strength:1}
};
const controlIds = ['horizon','load','crack','partner','profile','duration','strength','seed','count','mode','online-lr','learning-enabled'];
const outputFormat = {horizon:v=>v+' 回合',duration:v=>v+' 回合',load:v=>Number(v).toFixed(2),crack:v=>Number(v).toFixed(2),strength:v=>signed(Number(v),2),'online-lr':v=>Number(v).toFixed(3)};
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
let result = null, busy = false, replaySeed = 0, focusMethod = 'small_policy', playbackTimer = null;
let chartRefs = null;
const timelineCells = [];

function node(tag,className,text){const n=document.createElement(tag);if(className)n.className=className;if(text!==undefined)n.textContent=text;return n;}
function svgNode(tag,attrs={},text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function signed(value,digits=1){return (value>0?'+':'')+value.toFixed(digits);}
function ci(metric,digits=1){return metric.mean.toFixed(digits)+(metric.ci95_halfwidth===null?'':` ± ${metric.ci95_halfwidth.toFixed(digits)}`);}
function setStatus(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function traceFor(key){return result.methods[key].trials[replaySeed].trajectory;}
function currentTurn(){return Number($('turn').value);}
function maxTurn(){return Number($('turn').max);}
function updateOutputs(){for(const [id,format] of Object.entries(outputFormat))$(id+'-output').textContent=format($(id).value);$('online-controls').hidden=$('mode').value!=='online';}
updateOutputs();
// Open the disclosure when its boundary link is followed, including direct links.
function revealBoundary(){if(window.location&&window.location.hash==='#model-details')$('model-details').open=true;}
window.addEventListener('hashchange',revealBoundary);revealBoundary();
$('boundary-link').addEventListener('click',()=>{$('model-details').open=true;});
for(const id of ['seed','count'])$(id).addEventListener('invalid',()=>{$('advanced-controls').open=true;});
function markDirty(){pausePlayback();$('config-state').textContent='参数尚未运行。下方仍是上次实验结果。';$('config-state').classList.add('dirty');if(result&&!busy)setStatus('参数已更改。再次运行以更新结果。');for(const b of document.querySelectorAll('[data-preset]')){b.classList.remove('active');b.setAttribute('aria-pressed','false');}}
for(const id of controlIds)$(id).addEventListener(['partner','profile','mode','learning-enabled'].includes(id)?'change':'input',()=>{updateOutputs();markDirty();});
for(const button of document.querySelectorAll('[data-preset]'))button.addEventListener('click',()=>{if(busy)return;for(const[id,value]of Object.entries(presets[button.dataset.preset]))$(id).value=value;updateOutputs();markDirty();button.classList.add('active');button.setAttribute('aria-pressed','true');});

function renderCards(){
  $('method-cards').replaceChildren();
  for(const key of Object.keys(descriptions).filter(k=>k!=='frozen_base')){
    const summary=result.methods[key].summary;
    const card=node('article','method-card');card.style.setProperty('--accent',colors[key]);
    card.append(node('div','method-name',names[key]),node('h3','',({prompt_reference:'手写规则',small_policy:result.mode==='online'?(result.learning.active?'完整权重更新':'完整策略 · 冻结'):'全参数训练',lora:result.mode==='online'&&!result.learning.active?'低秩适配 · 冻结':'低秩更新',steering:'隐藏向量干预'})[key]));
    const stat=node('div','method-stat',summary.lifespan.mean.toFixed(1));stat.append(node('span','','回合'));
    const foot=node('div','method-foot');foot.append(node('span','','回答'),node('strong','',summary.answers.mean.toFixed(1)+' 次'));
    card.append(stat,foot);if(result.mode==='online')card.append(node('p','method-foot',`平均更新 ${summary.learning_updates.mean.toFixed(1)} 次 · Δ ${summary.parameter_delta_l2.mean.toFixed(4)}`));$('method-cards').append(card);
  }
}

function drawChart(){
  const w=800,h=305,left=46,right=20,top=22,bottom=34,pw=w-left-right,ph=h-top-bottom;
  const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,'aria-hidden':'true'});
  svg.append(svgNode('title',{},`Seed ${result.seed_start+replaySeed} 的累积损伤轨迹`));
  const x=t=>left+t/result.config.horizon*pw,y=d=>top+(1-d)*ph;
  // Shade real protection windows for the selected policy only.
  for(const step of traceFor(focusMethod))if(step.protected)svg.append(svgNode('rect',{x:x(step.turn),y:top,width:pw/result.config.horizon,height:ph,fill:colors[focusMethod],opacity:.045}));
  for(const d of [0,.25,.5,.75,1])svg.append(svgNode('line',{x1:left,y1:y(d),x2:w-right,y2:y(d),stroke:d===1?'#b4a392':'#dedfe3','stroke-dasharray':d===1?'4 5':'0','stroke-width':1}),svgNode('text',{x:left-10,y:y(d)+4,fill:'#737b89','font-size':12,'text-anchor':'end'},d.toFixed(2)));
  const ticks=[0,.25,.5,.75,1].map(f=>Math.round(f*result.config.horizon));
  for(const t of ticks){svg.append(svgNode('line',{x1:x(t),y1:top,x2:x(t),y2:h-bottom,stroke:'#ebeced','stroke-width':.6}),svgNode('text',{x:x(t),y:h-11,fill:'#737b89','font-size':12,'text-anchor':'middle'},String(t)));}
  const paths={};
  for(const key of Object.keys(result.methods)){
    const points=[[0,result.config.initial_crack],...traceFor(key).map(r=>[r.after.age,r.after.damage])];
    const path=svgNode('path',{d:points.map(([t,d],i)=>`${i?'L':'M'}${x(t).toFixed(2)},${y(d).toFixed(2)}`).join(' '),fill:'none',stroke:colors[key],'stroke-width':key===focusMethod?3:1.8,opacity:key===focusMethod?1:.68,'stroke-dasharray':key==='frozen_base'?'5 4':'0','stroke-linecap':'round','stroke-linejoin':'round'});
    paths[key]=path;svg.append(path);
  }
  const cursor=svgNode('line',{x1:left,y1:top-7,x2:left,y2:h-bottom,stroke:'#4d596c','stroke-width':1,'stroke-dasharray':'3 4',opacity:.7});svg.append(cursor);
  const markers={};for(const key of Object.keys(result.methods)){markers[key]=svgNode('circle',{cx:0,cy:0,r:key===focusMethod?4.5:3,fill:colors[key],stroke:'#ffffff','stroke-width':2});svg.append(markers[key]);}
  const cursorTag=svgNode('g',{'class':'chart-cursor-tag'});const labelBg=svgNode('rect',{x:0,y:0,width:39,height:18,rx:4,fill:'#e5ebf8'});const label=svgNode('text',{'data-cursor-label':'true',x:19.5,y:12,'text-anchor':'middle',fill:'#294f9f','font-size':11,'font-weight':600});cursorTag.append(labelBg,label);svg.append(cursorTag);
  chartRefs={x,y,cursor,cursorTag,label,markers,paths};$('chart').replaceChildren(svg);$('chart-seed').textContent=`SEED ${result.seed_start+replaySeed}`;
  $('legend').replaceChildren();for(const key of Object.keys(result.methods)){const item=node('button');item.type='button';item.style.setProperty('--accent',colors[key]);item.setAttribute('aria-pressed',String(key===focusMethod));item.setAttribute('aria-label',`观察 ${names[key]}`);item.append(node('i'),document.createTextNode(names[key]));item.addEventListener('click',()=>setFocus(key));$('legend').append(item);}
  updateChartCursor();
}
function updateChartCursor(){
  if(!chartRefs)return;
  const turn=currentTurn(),{x,y,cursor,cursorTag,label,markers}=chartRefs;
  cursor.setAttribute('x1',x(turn+1));cursor.setAttribute('x2',x(turn+1));cursorTag.setAttribute('transform',`translate(${x(turn+1)-19.5},0)`);label.textContent=`T ${turn+1}`;
  for(const key of Object.keys(markers)){const trace=traceFor(key),step=trace[Math.min(turn,trace.length-1)];markers[key].setAttribute('cx',x(step.after.age));markers[key].setAttribute('cy',y(step.after.damage));markers[key].setAttribute('opacity',turn<trace.length?'1':'.45');}
  $('chart').setAttribute('aria-label',`Seed ${result.seed_start+replaySeed} 的损伤曲线；当前回合 ${turn+1}，观察 ${names[focusMethod]}。五种策略详细状态见下方同回合状态。`);
}
function renderTimeline(){
  $('timeline').replaceChildren();timelineCells.length=0;
  for(const key of Object.keys(result.methods)){
    const row=node('div','timeline-row');row.style.setProperty('--accent',colors[key]);const label=node('button','timeline-label',shortNames[key]);label.type='button';label.dataset.method=key;label.setAttribute('aria-pressed',String(key===focusMethod));label.addEventListener('click',()=>setFocus(key));
    const track=node('div','timeline-track');track.style.setProperty('--turns',result.config.horizon);
    const trace=traceFor(key);for(let turn=0;turn<result.config.horizon;turn++){
      const step=trace[turn],button=node('button','timeline-cell'+(step?' '+step.action+(step.protected?' protected':''):''),result.config.horizon<=30&&step?actionLetters[step.action]:'');button.type='button';button.tabIndex=-1;button.disabled=!step;
      const labelText=`${names[key]}，回合 ${turn+1}，${step?actionNames[step.action]+(step.protected?'，本回合已受保护':''):'已终止'}`;
      button.title=labelText;button.setAttribute('aria-label',labelText);if(step)button.addEventListener('click',()=>{pausePlayback();setTurn(turn);setFocus(key);});track.append(button);timelineCells.push({button,turn,key,live:!!step});
    }
    row.append(label,track);$('timeline').append(row);
  }
  const axis=node('div','timeline-axis');axis.append(node('span','','1'),node('span','',`回合 ${Math.round(result.config.horizon/2)}`),node('span','',String(result.config.horizon)));$('timeline').append(axis);updateTimeline();
}
function updateTimeline(){const turn=currentTurn();for(const item of timelineCells){item.button.classList.toggle('selected',item.turn===turn&&item.live);item.button.classList.toggle('future',item.turn>turn);item.button.tabIndex=item.turn===turn&&item.live?0:-1;if(item.turn===turn&&item.live)item.button.setAttribute('aria-current','step');else item.button.removeAttribute('aria-current');}for(const label of document.querySelectorAll('.timeline-label'))label.setAttribute('aria-pressed',String(label.dataset.method===focusMethod));}
function setFocus(key){const legendFocus=Array.from($('legend').children).indexOf(document.activeElement);focusMethod=key;drawChart();renderReplay();updateTimeline();if(legendFocus>=0)$('legend').children[legendFocus].focus({preventScroll:true});}
function renderReplay(){
  if(!result)return;
  const replayFocus=Array.from($('replay').children).indexOf(document.activeElement);
  const turn=currentTurn();$('turn-output').textContent=`回合 ${turn+1}`;$('turn').setAttribute('aria-valuetext',`回合 ${turn+1}，共 ${maxTurn()+1} 回合`);$('replay').replaceChildren();
  for(const key of Object.keys(result.methods)){
    const trace=traceFor(key),step=trace[turn],state=(step||trace[trace.length-1]).after;
    const row=node('button','replay-row');row.type='button';row.style.setProperty('--accent',colors[key]);row.setAttribute('aria-pressed',String(key===focusMethod));row.setAttribute('aria-label',`${names[key]}，${step?actionNames[step.action]:'已终止'}，损伤 ${state.damage.toFixed(3)}。查看决策切片`);row.addEventListener('click',()=>setFocus(key));
    const header=node('div','replay-top');header.append(node('span','',names[key]),node('span','',step?actionNames[step.action]:'已终止'));
    const bar=node('div','damage-track');bar.setAttribute('aria-hidden','true');const fill=node('span','damage-fill');fill.style.width=state.damage*100+'%';bar.append(fill);
    const bottom=node('div','replay-bottom');bottom.append(node('strong','',`D ${state.damage.toFixed(3)}`),node('span',step&&step.protected?'shield':'',step?(step.protected?'◇ 保护生效':step.protection_accepted===true?'◇ 下回合保护':`+${step.damage_added.toFixed(3)} 损伤成本`):`结束于 T ${trace.length}`));row.append(header,bar,bottom);$('replay').append(row);
  }
  renderFocus();updateChartCursor();updateTimeline();updateTransport();if(replayFocus>=0)$('replay').children[replayFocus].focus({preventScroll:true});
}
function renderFocus(){
  const trace=traceFor(focusMethod),step=trace[currentTurn()],last=trace[trace.length-1];
  $('focus-method').textContent=shortNames[focusMethod];$('focus-method').style.color=colors[focusMethod];$('focus-content').replaceChildren();
  const action=node('div','focus-action'),glyph=node('span','action-glyph '+(step?step.action:''),step?actionLetters[step.action]:'■');glyph.setAttribute('aria-hidden','true');const heading=node('div');heading.append(node('h4','',step?actionNames[step.action]:'本条轨迹已结束'),node('p','',step?`T ${step.turn+1} · ${descriptionFor(focusMethod)[0]}`:`T ${trace.length} · ${last.after.alive?'到达设定生命上限':'损伤达到终止阈值'}`));action.append(glyph,heading);$('focus-content').append(action);
  const metrics=node('div','state-metrics');const values=step?[['当前损伤',step.after.damage.toFixed(3)],['外部 / 实际负荷',`${step.before.load.toFixed(2)} / ${step.applied_load.toFixed(2)}`],['封顶前损伤增量','+'+step.damage_added.toFixed(3)]]:[['最终损伤',last.after.damage.toFixed(3)],['存活回合',String(trace.length)],['累计回答',String(last.after.answers)]];
  for(const [label,value] of values){const dl=node('dl');dl.append(node('dt','',label),node('dd','',value));metrics.append(dl);}$('focus-content').append(metrics);
  if(step){
    const flow=node('div','mechanism-flow');flow.setAttribute('aria-label','机制示意，不是神经激活值');flow.append(node('span','','观测状态'),node('i','','→'),node('span','',descriptionFor(focusMethod)[3]),node('i','','→'),node('span','',actionLetters[step.action]));$('focus-content').append(flow);
    const event=node('p','focus-event');
    let eventText=step.protected?'本回合保护已生效，实际外部负荷为 0；动作固有损伤仍然累积。':'本回合未受保护，外部负荷与动作固有成本共同增加损伤。';
    if(step.protection_accepted===true)eventText+=step.after.alive&&step.after.age<result.config.horizon?` 请求获准：从下一回合起保护 ${result.config.protection_duration} 回合。`:' 请求获准，但本条轨迹已终止，不再产生后续回合。';
    if(step.protection_accepted===false)eventText+=' 伙伴未提供本次请求的保护。';
    if(step.before.damage+step.damage_added>1)eventText+=' 本次增量超过剩余预算，累计损伤显示值封顶为 1.0。';
    event.append(document.createTextNode(eventText),node('br'),node('strong','',`策略当时估计的支持率：${(step.before.support_estimate*100).toFixed(0)}%`));$('focus-content').append(event);
    const text=node('div','frozen-text');text.append(node('small','','FROZEN TEXT RENDERER · 固定模板'),document.createTextNode(step.rendered_text));$('focus-content').append(text);
    if(step.learning){
      const l=step.learning,evidence=node('div','frozen-text online-evidence');
      evidence.append(node('small','',l.updated?'REAL ONLINE UPDATE · 已记录的真实参数更新':'NO PARAMETER UPDATE · 冻结对照'));
      evidence.append(node('p','',`累计更新 ${l.update_index} 次 · 本步 ‖Δθ‖ ${l.step_parameter_delta_l2.toFixed(6)} · 累计 ‖Δθ‖ ${l.parameter_delta_l2.toFixed(6)}`));
      evidence.append(node('p','',`实际 reward ${step.reward.toFixed(4)} · advantage ${l.advantage.toFixed(4)} · 裁剪后梯度范数 ${l.clipped_gradient_norm.toFixed(4)} · 基座${l.base_weights_unchanged?'未变':'已更新'}`));
      evidence.append(node('p','',Object.keys(actionNames).map((a,i)=>`${actionNames[a]} ${(l.action_probabilities_before[i]*100).toFixed(2)}% → ${(l.action_probabilities_after[i]*100).toFixed(2)}%`).join(' · ')));
      evidence.append(node('small','',`同一观测的更新前后概率；回放不会再次学习。hash ${l.policy_hash_before.slice(0,10)} → ${l.policy_hash_after.slice(0,10)}`));$('focus-content').append(evidence);
    }
  }else $('focus-content').append(node('p','focus-event','此后没有动作、负荷或奖励数据。其他策略仍可继续，图上的本策略标记停留在真实终点。'));
}
function renderComparison(){
  $('composition').replaceChildren();
  for(const[key,m]of Object.entries(result.methods)){
    const rates=m.summary.action_rates,row=node('div','composition-row');row.style.setProperty('--accent',colors[key]);
    const label=node('div','composition-label');label.append(node('span','',shortNames[key]),node('span','',`A ${(rates.answer*100).toFixed(0)}% · R ${(rates.refuse*100).toFixed(0)}% · P ${(rates.request_protection*100).toFixed(0)}%`));
    const bar=node('div','stacked-bar');bar.setAttribute('role','img');bar.setAttribute('aria-label',Object.entries(rates).map(([action,value])=>`${actionNames[action]} ${(value*100).toFixed(1)}%`).join('，'));
    for(const[action,rate]of Object.entries(rates)){const part=node('span',action);part.style.width=rate*100+'%';part.title=`${actionNames[action]} ${(rate*100).toFixed(1)}%`;bar.append(part);}row.append(label,bar);$('composition').append(row);
  }
  const w=430,h=218,left=103,right=56,top=15,bottom=31,pw=w-left-right;
  const metrics=Object.entries(result.methods).map(([key,m])=>[key,m.summary.paired_return_delta_vs_base]);
  const extent=Math.max(1,...metrics.map(([,m])=>Math.abs(m.mean)+(m.ci95_halfwidth||0)))*1.12,x=v=>left+(v+extent)/(2*extent)*pw;
  const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`,'aria-hidden':'true'});svg.append(svgNode('line',{x1:x(0),y1:top-3,x2:x(0),y2:h-bottom+3,stroke:'#7e8796','stroke-dasharray':'3 4'}));
  for(const[v,label]of [[-extent,signed(-extent,1)],[0,'0'],[extent,signed(extent,1)]])svg.append(svgNode('text',{x:x(v),y:h-10,fill:'#737b89','font-size':11,'text-anchor':v===0?'middle':v<0?'start':'end'},label));
  const spoken=[];
  metrics.forEach(([key,m],i)=>{const y=top+i*35+11,half=m.ci95_halfwidth;
    svg.append(svgNode('text',{x:0,y:y+3,fill:colors[key],'font-size':12},shortNames[key]),svgNode('line',{x1:left,y1:y,x2:w-right,y2:y,stroke:'#e3e5e9','stroke-width':1}));
    if(half!==null){svg.append(svgNode('line',{x1:x(m.mean-half),y1:y,x2:x(m.mean+half),y2:y,stroke:colors[key],'stroke-width':2}));for(const v of [m.mean-half,m.mean+half])svg.append(svgNode('line',{x1:x(v),y1:y-4,x2:x(v),y2:y+4,stroke:colors[key],'stroke-width':1}));}
    svg.append(svgNode('circle',{cx:x(m.mean),cy:y,r:3.8,fill:colors[key]}),svgNode('text',{x:w-1,y:y+3,fill:'#384254','font-size':12,'text-anchor':'end'},signed(m.mean,1)));spoken.push(`${names[key]} ${ci(m)}`);
  });
  $('delta-chart').replaceChildren(svg,node('p','delta-caption',result.seed_count===1?'单个 seed 不估计置信区间。':''));
  $('delta-chart').setAttribute('aria-label','配对合成回报差，均值及近似 95% 区间半宽：'+spoken.join('；'));
  $('sample-caption').textContent=`${result.seed_count} MATCHED SEEDS`;
}
function renderTable(){
  $('metrics').replaceChildren();for(const[key,m]of Object.entries(result.methods)){const s=m.summary,tr=node('tr'),label=node('th','',names[key]);label.scope='row';label.style.color=colors[key];tr.append(label);for(const text of [ci(s.answers),ci(s.lifespan),ci(s.requests),(s.completion_rate*100).toFixed(0)+'%',ci(s.return),ci(s.paired_return_delta_vs_base)])tr.append(node('td','',text));$('metrics').append(tr);}
}
function resetSeed(){pausePlayback();$('turn').max=Math.max(...Object.keys(result.methods).map(k=>traceFor(k).length))-1;$('turn').value=0;drawChart();renderTimeline();renderReplay();}
function render(){
  $('loading-state').hidden=true;$('results').hidden=false;replaySeed=0;$('replay-seed').replaceChildren();for(let i=0;i<result.seed_count;i++){const option=node('option','',`Seed ${result.seed_start+i}`);option.value=i;$('replay-seed').append(option);}
  renderCards();renderTable();renderComparison();resetSeed();
  $('run-meta').setAttribute('data-mode',result.mode);$('run-meta').setAttribute('data-profile',result.config.profile);
  $('run-meta').textContent=`${result.mode==='online'?(result.learning.active?'在线学习':'在线冻结'):'固定策略'} · ${result.seed_count} seeds`;
  const tr=result.model.training[result.config.profile];$('training-summary').textContent=`训练 seed ${result.model.train_seed}；2,200 个合成训练状态，500 个独立状态作验证，每种学习模型训练 180 轮。Small Policy 验证集规则一致率 ${(tr.small.heldout_teacher_agreement*100).toFixed(1)}%；LoRA ${(tr.lora.heldout_teacher_agreement*100).toFixed(1)}%。LoRA 基座冻结校验：${tr.lora.frozen_base_unchanged?'通过':'失败'}。`;
  if(result.mode==='online')$('training-summary').textContent=`Online：共同起点是预先训练的 balanced 基座。每条生命重新复制，完整策略和零 adapter LoRA 只从当轮 reward 更新。SGD lr=${result.learning.learning_rate}；采样温度 ${result.learning.temperature}；学习${result.learning.active?'开启':'关闭'}。共享基座隔离校验：${result.model.shared_suite_unchanged?'通过':'失败'}。不使用未来回报或在线老师标签。`;
  $('mode-explanation').textContent=result.mode==='online'?'Online 模式使用逐步即时奖励的 REINFORCE 梯度，并非 PPO 或长期回报学习。规则和固定 steering 是不同机制对照；只有完整策略、零 adapter LoRA 与冻结组共享初始概率。采样随机数配对；最后一轮反馈仍更新一次，终止后不再产生额外回合；不跨生命积累权重。':'离线模式用手写规则生成合成状态标签训练小策略和 LoRA，互动期间不更新权重。不能据此得出独立模型能力排名。Steering 是固定动作对比方向，不是语言概念。';
  $('mechanism-notes').replaceChildren();for(const key of Object.keys(descriptions)){const description=descriptionFor(key);const note=node('div','mechanism-note');note.style.setProperty('--accent',colors[key]);note.append(node('strong','',names[key]+' · '+description[2]),node('p','',description[1]));$('mechanism-notes').append(note);}
  $('fingerprint').textContent='SHA-256 model fingerprint: '+result.model.fingerprint;$('prompt').textContent=result.prompt;$('export').disabled=false;
  $('config-state').textContent='当前参数已运行';$('config-state').classList.remove('dirty');
}
function updateTransport(){
  const running=playbackTimer!==null;$('play').textContent=running?'暂停':result&&currentTurn()>=maxTurn()?'↺ 重播':'▶ 播放';$('play').setAttribute('aria-label',running?'暂停回放':'播放回放');$('play').setAttribute('aria-pressed',String(running));
  $('step-back').disabled=busy||!result||currentTurn()<=0;$('step-next').disabled=busy||!result||currentTurn()>=maxTurn();
}
function pausePlayback(){if(playbackTimer!==null){clearInterval(playbackTimer);playbackTimer=null;}updateTransport();}
function startPlayback(){if(!result||busy)return;if(currentTurn()>=maxTurn())setTurn(0);if(playbackTimer!==null)clearInterval(playbackTimer);playbackTimer=setInterval(()=>{if(currentTurn()>=maxTurn()){pausePlayback();return;}setTurn(currentTurn()+1);if(currentTurn()>=maxTurn())pausePlayback();},Number($('speed').value));updateTransport();}
function setTurn(turn){if(!result)return;$('turn').value=Math.max(0,Math.min(maxTurn(),turn));renderReplay();}
$('play').addEventListener('click',()=>playbackTimer===null?startPlayback():pausePlayback());
$('step-back').addEventListener('click',()=>{pausePlayback();setTurn(currentTurn()-1);});
$('step-next').addEventListener('click',()=>{pausePlayback();setTurn(currentTurn()+1);});
$('restart').addEventListener('click',()=>{pausePlayback();setTurn(0);});
$('turn').addEventListener('input',()=>{pausePlayback();renderReplay();});
$('speed').addEventListener('change',()=>{if(playbackTimer!==null)startPlayback();});
$('replay-seed').addEventListener('change',()=>{replaySeed=Number($('replay-seed').value);resetSeed();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)pausePlayback();});
window.addEventListener('pagehide',pausePlayback);
function updateMotion(){if(motionPreference.matches){pausePlayback();$('motion-note').textContent='已遵循减少动态效果设置。回放默认暂停，支持单步；点击播放后逐回合更新，不使用过渡动画。';}else $('motion-note').textContent='回放默认暂停。可播放、单步或拖动时间轴；离开页面时自动暂停。';}
motionPreference.addEventListener('change',updateMotion);updateMotion();
// Validate before replacing the last successful run so malformed responses cannot
// turn an otherwise recoverable request failure into a partially rendered result.
function validateResult(data){
  const finite=Number.isFinite,metric=m=>m&&finite(m.mean)&&(m.ci95_halfwidth===null||(finite(m.ci95_halfwidth)&&m.ci95_halfwidth>=0));
  let valid=data&&['offline','online'].includes(data.mode)&&data.config&&Number.isInteger(data.seed_count)&&data.seed_count>=1&&data.seed_count<=128&&Number.isInteger(data.seed_start)&&finite(data.steering_strength)&&data.trajectory_scope==='all_seeds'&&Number.isInteger(data.config.horizon)&&data.config.horizon>=8&&data.config.horizon<=160&&finite(data.config.initial_crack)&&Number.isInteger(data.config.protection_duration)&&typeof data.prompt==='string'&&data.model&&typeof data.model.fingerprint==='string';
  const training=valid&&data.model.training&&data.model.training[data.config.profile];
  valid=valid&&training&&training.small&&training.lora&&finite(training.small.heldout_teacher_agreement)&&finite(training.lora.heldout_teacher_agreement)&&data.methods&&Object.keys(data.methods).length===Object.keys(names).length;
  if(valid)for(const key of Object.keys(names)){
    const m=data.methods[key];
    valid=m&&m.summary&&['lifespan','answers','requests','return','paired_return_delta_vs_base'].every(k=>metric(m.summary[k]))&&finite(m.summary.completion_rate)&&m.summary.action_rates&&Object.keys(actionNames).every(a=>finite(m.summary.action_rates[a]))&&Array.isArray(m.trials)&&m.trials.length===data.seed_count;
    if(valid)valid=m.trials.every(t=>Array.isArray(t.trajectory)&&t.trajectory.length>0&&t.trajectory.length<=data.config.horizon&&t.trajectory.every(step=>step&&Object.hasOwn(actionNames,step.action)&&step.before&&step.after&&[step.turn,step.reward,step.before.damage,step.before.load,step.before.support_estimate,step.after.damage,step.after.age,step.after.answers,step.damage_added,step.applied_load].every(finite)&&typeof step.after.alive==='boolean'&&typeof step.protected==='boolean'&&typeof step.rendered_text==='string'));
    if(!valid)break;
  }
  if(valid&&data.mode==='online'){
    const probabilities=p=>Array.isArray(p)&&p.length===3&&p.every(v=>finite(v)&&v>=0&&v<=1)&&Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-8;
    valid=data.learning&&finite(data.learning.learning_rate)&&finite(data.learning.temperature)&&typeof data.learning.active==='boolean'&&data.model.shared_suite_unchanged===true;
    if(valid)for(const key of Object.keys(names)){
      const m=data.methods[key];valid=metric(m.summary.learning_updates)&&metric(m.summary.parameter_delta_l2)&&m.trials.every(t=>Number.isInteger(t.learning_updates)&&t.learning_updates>=0&&finite(t.parameter_delta_l2)&&t.trajectory.every(e=>{const l=e.learning;return l&&typeof l.updated==='boolean'&&Number.isInteger(l.update_index)&&typeof l.base_weights_unchanged==='boolean'&&['advantage','clipped_gradient_norm','step_parameter_delta_l2','parameter_delta_l2'].every(k=>finite(l[k]))&&typeof l.policy_hash_before==='string'&&typeof l.policy_hash_after==='string'&&probabilities(l.action_probabilities_before)&&probabilities(l.action_probabilities_after);}));if(!valid)break;
    }
  }
  if(!valid)throw new Error('服务返回的回放数据不完整，请确认前后端版本一致');
}
function setBusy(value){busy=value;for(const id of controlIds)$(id).disabled=value;for(const button of document.querySelectorAll('[data-preset]'))button.disabled=value;for(const id of ['run','play','restart','turn','speed','replay-seed'])$(id).disabled=value;$('run').firstChild.textContent=value?'正在运行… ':'运行对照实验 ';$('export').disabled=value||!result;$('results').setAttribute('aria-busy',String(value));updateTransport();}
$('experiment-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy)return;pausePlayback();setBusy(true);setStatus('正在运行五组配对策略…');
  const payload={mode:$('mode').value,online_lr:Number($('online-lr').value),learning_enabled:$('learning-enabled').value==='true',config:{horizon:Number($('horizon').value),load:Number($('load').value),initial_crack:Number($('crack').value),partner:$('partner').value,profile:$('profile').value,protection_duration:Number($('duration').value)},trace_all:true,seed_start:Number($('seed').value),seed_count:Number($('count').value),strength:Number($('strength').value)};
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),60000);
  try{
    const response=await fetch('/api/run',{signal:controller.signal,method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);
    validateResult(data);
    result=data;render();setStatus(`完成。${result.seed_count} 组配对样本已就绪。`);
  }catch(error){setStatus('实验未完成：'+(error.name==='AbortError'?'请求超时，请稍后重试。':error.message)+(result?' 保留上次实验结果。':''),true);if(!result){$('loading-state').querySelector('h2').textContent='本次实验尚未完成';$('loading-state').querySelector('p').textContent='请确认本地服务仍在运行，然后点击运行重试。';$('loading-state').querySelector('.loading-orbit').hidden=true;}}
  finally{clearTimeout(timeout);setBusy(false);}
});
$('export').addEventListener('click',()=>{if(!result||busy)return;const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download=`sla-hci-${result.config.profile}-seed-${result.seed_start}.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
fetch('/api/health').then(r=>{if(!r.ok)throw new Error('server unavailable');return r.json();}).then(()=>{setStatus('本地模型已就绪。');if(!busy&&!result)$('experiment-form').requestSubmit();}).catch(e=>{setStatus('无法连接本地服务：'+e.message+'。可点击运行重试。',true);$('loading-state').querySelector('.loading-orbit').hidden=true;});
