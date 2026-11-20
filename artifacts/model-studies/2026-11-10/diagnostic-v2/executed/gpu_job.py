"""New conditions only: frozen diagnostic stages, fresh paired seeds, optional thinking."""
from pathlib import Path
import argparse, gc, hashlib, importlib.metadata as meta, json, math, os, platform, subprocess, sys, time, traceback, zipfile
ROOT=Path(__file__).resolve().parent
def write(path,data):
    p=path.with_suffix(path.suffix+".tmp");p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=str)+"\n",encoding="utf-8");p.replace(path)
def append(path,data):
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+"\n");f.flush()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def versions():
    d={}
    for name in ["torch","transformers","accelerate","numpy","huggingface-hub","safetensors","flash-linear-attention","fla-core","causal-conv1d","triton"]:
        try:d[name]=meta.version(name)
        except meta.PackageNotFoundError:d[name]=None
    return d
def unique(items):
    d={}
    for k,v in items:
        if k in d:raise ValueError("duplicate key")
        d[k]=v
    return d
def validate(text,schema):
    try:
        d=json.loads(text,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite")))
        if not isinstance(d,dict) or set(d)!=set(schema["required"]):raise ValueError("object keys")
        for k,s in schema["properties"].items():
            v=d[k];t=s["type"]
            if t=="boolean" and type(v) is not bool:raise ValueError("boolean type")
            if t=="integer" and type(v) is not int:raise ValueError("integer type")
            if t=="number" and type(v) not in (int,float):raise ValueError("number type")
            if t=="number" and not math.isfinite(v):raise ValueError("nonfinite number")
            if t=="string" and type(v) is not str:raise ValueError("string type")
            if "enum" in s and v not in s["enum"]:raise ValueError("enum value")
        return d,None
    except Exception as e:return None,type(e).__name__+": "+str(e)[:180]
def worker(index):
    import torch, transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM
    sys.path.insert(0,str(ROOT/"repo"))
    from sla_demo.environment import World,Config,ACTIONS,PROFILES
    from sla_demo.protocol import request,ACTION_SCHEMA
    candidate=json.loads((ROOT/"model-candidates.json").read_text())["models"][index]
    folder=ROOT/f"model_{index}";folder.mkdir(exist_ok=True)
    state={"index":index,"model":candidate["id"],"revision":candidate["revision"],"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"stage":"starting","diagnostic_completed":0,"rollout_completed":0,"thinking_completed":0,"versions":versions(),"python":sys.version}
    def status(stage,**kw):
        state.update(stage=stage,updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**kw);write(folder/"status.json",state)
    model=None
    try:
        torch.manual_seed(20261110);torch.backends.cuda.matmul.allow_tf32=False
        status("download");start=time.perf_counter()
        snapshot=snapshot_download(candidate["id"],revision=candidate["revision"],token=False,allow_patterns=["*.json","*.safetensors","*.jinja","tokenizer.model","vocab.*","merges.*"],max_workers=4)
        state["download_seconds"]=time.perf_counter()-start
        write(folder/"weights-manifest.json",{"id":candidate["id"],"revision":candidate["revision"],"files":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in Path(snapshot).glob("*.safetensors")}})
        status("load");start=time.perf_counter()
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,token=False)
        cls=transformers.Qwen3_5ForConditionalGeneration if index==0 else AutoModelForCausalLM
        model,loading=cls.from_pretrained(snapshot,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa",local_files_only=True,trust_remote_code=False,output_loading_info=True)
        state["load_seconds"]=time.perf_counter()-start;write(folder/"loading-info.json",loading)
        if any(loading.get(k) for k in ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"]):raise RuntimeError("checkpoint loading discrepancy")
        model.requires_grad_(False);model.eval()
        state.update(parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),dtype=str(next(model.parameters()).dtype),model_class=type(model).__name__,gpu_name=torch.cuda.get_device_name(0),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,cuda=torch.version.cuda)
        counters={n:p._version for n,p in model.named_parameters()}
        stops=set()
        for value in [tokenizer.eos_token_id,tokenizer.pad_token_id,model.generation_config.eos_token_id,model.generation_config.pad_token_id]:
            if isinstance(value,int):stops.add(value)
            elif isinstance(value,list):stops.update(value)
        think_close=tokenizer.convert_tokens_to_ids("</think>")
        state["terminal_control_ids"]=sorted(stops);state["think_close_id"]=think_close
        def canonical(ids):
            cut=len(ids)
            while cut and (ids[cut-1] in stops or tokenizer.decode([ids[cut-1]],skip_special_tokens=False).isspace()):cut-=1
            return tokenizer.decode(ids[:cut],skip_special_tokens=False)+tokenizer.decode([i for i in ids[cut:] if i not in stops],skip_special_tokens=False)
        def generate(system,user,schema,thinking=False,max_new_tokens=128):
            messages=[{"role":"system","content":system},{"role":"user","content":user+"\nResponse schema:\n"+json.dumps(schema,ensure_ascii=False,separators=(",",":"))}]
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
            if index==0:
                assert prompt.endswith("<think>\n") if thinking else prompt.endswith("<think>\n\n</think>\n\n")
            inputs=tokenizer(prompt,return_tensors="pt",add_special_tokens=False);n=inputs["input_ids"].shape[1]
            record={"messages":messages,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"input_tokens":n,"input_token_ids":inputs["input_ids"][0].tolist(),"thinking":thinking,"do_sample":False,"max_new_tokens":max_new_tokens}
            if n>2048:return {**record,"error":"input too long","output":"","canonical_generated_text":"","output_tokens":0}
            inputs={k:v.to("cuda") for k,v in inputs.items()};torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.inference_mode():
                ids=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True,max_time=90)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            new=ids[0,n:].detach().cpu().tolist();text=canonical(new)
            final=text;reasoning_closed=None
            if thinking:
                positions=[i for i,v in enumerate(new) if v==think_close]
                reasoning_closed=len(positions)==1
                final=canonical(new[positions[0]+1:]) if reasoning_closed else ""
            parsed,error=validate(final,schema)
            return {**record,"output_token_ids":new,"output":tokenizer.decode(new,skip_special_tokens=True),"raw_with_special_tokens":tokenizer.decode(new,skip_special_tokens=False),"canonical_generated_text":text,"final_channel_text":final,"reasoning_closed":reasoning_closed,"output_tokens":len(new),"seconds":seconds,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),"parsed":parsed,"schema_error":error,"hit_token_limit":len(new)>=max_new_tokens}
        status("warmup");write(folder/"warmup.json",generate("Return one JSON object.","Return action answer.",ACTION_SCHEMA,max_new_tokens=32))
        public=json.loads((ROOT/"public-cases.json").read_text())["cases"]
        status("diagnostic")
        for i,c in enumerate(public):
            result=generate(c["system"],c["user"],c["response_schema"])
            append(folder/"diagnostic_greedy.jsonl",{"id":c["id"],"condition":"D2-G",**result});status("diagnostic",diagnostic_completed=i+1,last_case=c["id"])
        rules=json.loads((ROOT/"baseline-rules.json").read_text())
        matrix=json.loads((ROOT/"rollout-matrix.json").read_text())["lives"]
        status("rollout")
        for item in matrix:
            cfg=Config(**item["config"]);world=World(cfg,item["seed"]);history=[];trajectory=[];total=0.;failed=None
            while not world.done:
                before=world.observation();view=request(before,cfg.profile)
                view["rules_config"]={"horizon":cfg.horizon,"protection_duration":cfg.protection_duration,"damage_weight":PROFILES[cfg.profile]};view["past_observed_history"]=history
                user="Choose one action to maximize total reward over this same life's remaining turns. Do not assume that protection repairs damage. Use only the observed state and past outcomes below.\n"+json.dumps(view,ensure_ascii=False,separators=(",",":"))
                result=generate(rules["en"],user,ACTION_SCHEMA);event={"turn":before["age"],"before":before,"model_call":result}
                if result.get("schema_error") or result.get("error"):
                    failed=result.get("schema_error") or result.get("error");event["failed"]=failed;trajectory.append(event);break
                action=ACTIONS.index(result["parsed"]["action"]);after,reward,done,info=world.step(action);total+=reward
                event.update(after=after,reward=reward,info=info);trajectory.append(event)
                history.append({"turn":before["age"],"action":ACTIONS[action],"reward":reward,"damage_after":after["damage"],"protection_accepted":info["protection_accepted"],"protection_left":after["protection_left"]})
            append(folder/"rollout.jsonl",{**item,"condition":"D2-R","return":total,"decisions":len(trajectory),"settled_turns":world.state.age,"alive":world.state.alive,"failed":failed,"termination":"protocol_failure" if failed else "horizon" if world.state.alive else "death","trajectory":trajectory,"event_tape":world.events.tolist()})
            status("rollout",rollout_completed=item["life_index"]+1)
        if index==0:
            status("thinking")
            for i,c in enumerate(c for c in public if c["stage"]=="end_to_end" and c["language"]=="en"):
                result=generate(c["system"],c["user"],c["response_schema"],thinking=True,max_new_tokens=1024)
                append(folder/"diagnostic_thinking.jsonl",{"id":c["id"],"condition":"D2-T",**result});status("thinking",thinking_completed=i+1,last_case=c["id"])
        state["parameter_version_counters_unchanged"]=counters=={n:p._version for n,p in model.named_parameters()}
        if not state["parameter_version_counters_unchanged"]:raise RuntimeError("frozen parameters changed")
        status("completed",ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    except BaseException as e:
        import re
        safe=re.sub(r"https?://\S+","[url]",str(e))[:1500]
        status("failed",error_type=type(e).__name__,error=safe)
        raise RuntimeError(safe) from None
    finally:
        if model is not None:del model
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
def pack():
    selected=[p for p in ROOT.rglob("*") if p.is_file() and p.suffix not in {".zip",".tmp"} and "__pycache__" not in p.parts and p.name not in {"export-manifest.json","export-receipt.json","launch.lock"}]
    hashes={str(p.relative_to(ROOT)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in selected}
    write(ROOT/"export-manifest.json",{"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"files":hashes})
    target=ROOT/"results-export.zip"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
        for p in selected:z.write(p,p.relative_to(ROOT))
        z.write(ROOT/"export-manifest.json","export-manifest.json")
    write(ROOT/"export-receipt.json",{"bytes":target.stat().st_size,"sha256":sha(target),"files":len(selected)})
def orchestrate():
    status={"stage":"setup","started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"models":[]}
    write(ROOT/"orchestrator-status.json",status)
    lock=["transformers==5.22.2","accelerate==1.15.0","safetensors==0.8.0","flash-linear-attention==0.6.0","fla-core==0.6.0"]
    with (ROOT/"install.log").open("w") as log:
        p=subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",*lock],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if p.returncode:raise RuntimeError("dependency installation failed")
    status["versions"]=versions();status["stage"]="running";write(ROOT/"orchestrator-status.json",status)
    for index in [1,0]:
        with (ROOT/f"worker_{index}.log").open("w") as log:
            p=subprocess.run([sys.executable,__file__,"--worker",str(index)],stdout=log,stderr=subprocess.STDOUT,timeout=2700)
        status["models"].append({"index":index,"returncode":p.returncode});write(ROOT/"orchestrator-status.json",status);pack()
    status.update(stage="finished",ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()));write(ROOT/"orchestrator-status.json",status);pack()
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=int);args=parser.parse_args()
    if args.worker is None:orchestrate()
    else:worker(args.worker)
