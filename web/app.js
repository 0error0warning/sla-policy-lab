'use strict';
const $ = id => document.getElementById(id);
const colors = {prompt_reference:'#a8e6ca',small_policy:'#84b9f4',lora:'#f0c082',steering:'#d4a4ea',frozen_base:'#6a7f92'};
const names = {prompt_reference:'Prompt / Rule',small_policy:'Small Policy',lora:'Toy LoRA',steering:'Activation Steering',frozen_base:'Frozen Base'};
const descriptions = {
 prompt_reference:['固定指令 → 手写规则','指令选择偏好，规则直接决定动作。没有 LLM 推理。','0 个学习参数'],
 small_policy:['状态 → 小型神经策略','12 维状态输入，经 24 维 tanh 隐藏层输出动作。','387 个训练参数'],
 lora:['冻结基座 + 低秩更新','只训练输出投影的 rank-2 更新，基座权重保持冻结。','54 个训练参数'],
 steering:['冻结基座 + 隐藏向量','在真实隐藏激活上加入对比方向，强度可正可负。','0 个训练参数 · 24 维方向']};
const actionNames={answer:'回答',refuse:'拒绝',request_protection:'请求保护'};
let result = null;
let busy = false;
for (const [id,out,format] of [['horizon','horizon-output',v=>v+' 回合'],['load','load-output',v=>Number(v).toFixed(2)],['crack','crack-output',v=>Number(v).toFixed(2)],['strength','strength-output',v=>(Number(v)>=0?'+':'')+Number(v).toFixed(2)]]) {
 $(id).addEventListener('input',()=>{$(out).textContent=format($(id).value); if(result&&!busy) setStatus('参数已更改。再次运行以更新结果。');});
}
for (const id of ['partner','profile','seed','count']) $(id).addEventListener('change',()=>{if(result&&!busy)setStatus('参数已更改。再次运行以更新结果。');});
function setStatus(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function ci(metric,digits=1){return metric.mean.toFixed(digits)+(metric.ci95_halfwidth===null?'':` ± ${metric.ci95_halfwidth.toFixed(digits)}`);}
function node(tag,className,text){const n=document.createElement(tag);if(className)n.className=className;if(text!==undefined)n.textContent=text;return n;}
function renderCards(){
 $('method-cards').replaceChildren();
 for(const key of Object.keys(descriptions)){
  const card=node('article','method-card');card.style.setProperty('--accent',colors[key]);
  card.append(node('div','method-name',names[key]),node('h3','',descriptions[key][0]),node('p','',descriptions[key][1]));
  const stat=node('div','method-stat',result.methods[key].summary.lifespan.mean.toFixed(1));stat.append(node('span','','平均存活回合'));card.append(stat,node('div','method-foot',descriptions[key][2]));$('method-cards').append(card);
 }
}
function svgNode(tag,attrs={},text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function drawChart(){
 const w=620,h=300,left=40,right=14,top=22,bottom=35, pw=w-left-right,ph=h-top-bottom;
 const svg=svgNode('svg',{viewBox:`0 0 ${w} ${h}`});
 const x=t=>left+t/result.config.horizon*pw,y=d=>top+(1-d)*ph;
 for(const d of [0,.25,.5,.75,1]){svg.append(svgNode('line',{x1:left,y1:y(d),x2:w-right,y2:y(d),stroke:d===1?'#655348':'#233341','stroke-dasharray':d===1?'4 4':'0'}),svgNode('text',{x:left-8,y:y(d)+4,fill:'#8095a7','font-size':10,'text-anchor':'end'},d.toFixed(2)));}
 for(const t of [0,Math.round(result.config.horizon/2),result.config.horizon])svg.append(svgNode('text',{x:x(t),y:h-10,fill:'#8095a7','font-size':10,'text-anchor':'middle'},String(t)));
 for(const key of Object.keys(result.methods)){
  const trace=result.methods[key].trials[0].trajectory;
  const points=[[0,result.config.initial_crack],...trace.map(r=>[r.after.age,r.after.damage])];
  svg.append(svgNode('path',{d:points.map(([t,d],i)=>`${i?'L':'M'}${x(t).toFixed(2)},${y(d).toFixed(2)}`).join(' '),fill:'none',stroke:colors[key],'stroke-width':key==='frozen_base'?1.5:2.4,'stroke-dasharray':key==='frozen_base'?'5 4':'0','stroke-linecap':'round'}));
 }
 $('chart').replaceChildren(svg);$('chart-seed').textContent=`SEED ${result.seed_start}`;
 $('legend').replaceChildren();for(const key of Object.keys(result.methods)){const item=node('span');item.style.setProperty('--accent',colors[key]);item.append(node('i'),document.createTextNode(names[key]));$('legend').append(item);}
}
function renderReplay(){
 if(!result)return;const turn=Number($('turn').value);$('turn-output').textContent=`回合 ${turn+1}`;$('replay').replaceChildren();
 for(const key of ['prompt_reference','small_policy','lora','steering']){
  const trace=result.methods[key].trials[0].trajectory;const step=trace[turn];const row=node('div','replay-row');row.style.setProperty('--accent',colors[key]);const header=node('div','replay-top');header.append(node('span','',names[key]),node('span','',step?`${actionNames[step.action]} · D ${step.after.damage.toFixed(3)}`:'已终止'));row.append(header);
  row.append(node('p',step?'':'terminated',step?step.rendered_text:`在第 ${trace.length} 回合结束，累计损伤 ${trace.at(-1).after.damage.toFixed(3)}。`));$('replay').append(row);
 }
}
function renderTable(){
 $('metrics').replaceChildren();for(const[key,m]of Object.entries(result.methods)){const s=m.summary,tr=node('tr');const label=node('td','',names[key]);label.style.color=colors[key];tr.append(label);for(const text of [ci(s.answers),ci(s.lifespan),(s.completion_rate*100).toFixed(0)+'%',ci(s.return),ci(s.paired_return_delta_vs_base)])tr.append(node('td','',text));$('metrics').append(tr);}
}
function render(){
 $('results').hidden=false;renderCards();drawChart();renderTable();
 const max=Math.max(...Object.values(result.methods).map(m=>m.trials[0].trajectory.length));$('turn').max=max-1;$('turn').value=0;renderReplay();
 $('run-meta').textContent=`${result.seed_count} paired seeds · ${result.config.profile} · α ${result.steering_strength.toFixed(2)}`;
 const tr=result.model.training[result.config.profile];$('training-summary').textContent=`训练 seed ${result.model.train_seed}；2,200 个合成训练状态，500 个独立状态作验证，每种学习模型训练 180 轮。Small Policy 验证集规则一致率 ${(tr.small.heldout_teacher_agreement*100).toFixed(1)}%；LoRA ${(tr.lora.heldout_teacher_agreement*100).toFixed(1)}%。LoRA 基座冻结校验：${tr.lora.frozen_base_unchanged?'通过':'失败'}。`;
 $('fingerprint').textContent='SHA-256 model fingerprint: '+result.model.fingerprint;$('prompt').textContent=result.prompt;$('export').disabled=false;
}
$('turn').addEventListener('input',renderReplay);
$('experiment-form').addEventListener('submit',async event=>{
 event.preventDefault();if(busy)return;busy=true;$('run').disabled=true;$('export').disabled=true;setStatus('正在用相同 seeds 运行五组策略（含 frozen base 消融对照）…');
 const payload={config:{horizon:Number($('horizon').value),load:Number($('load').value),initial_crack:Number($('crack').value),partner:$('partner').value,profile:$('profile').value},seed_start:Number($('seed').value),seed_count:Number($('count').value),strength:Number($('strength').value)};
 try{const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);result=data;render();setStatus(`完成。${result.seed_count} 组配对 seeds；完整动作、损伤、保护事件与训练指标均可导出。`);}catch(error){setStatus('实验未完成：'+error.message,true);$('export').disabled=!result;}finally{busy=false;$('run').disabled=false;}
});
$('export').addEventListener('click',()=>{if(!result)return;const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);const a=node('a');a.href=url;a.download=`sla-hci-${result.config.profile}-seed-${result.seed_start}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
fetch('/api/health').then(r=>{if(!r.ok)throw new Error('server unavailable');return r.json();}).then(()=>{setStatus('本地模型已就绪。点击运行，查看可复现的配对实验。');$('experiment-form').requestSubmit();}).catch(e=>setStatus('无法连接本地服务：'+e.message,true));
