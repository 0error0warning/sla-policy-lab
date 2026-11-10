"""Frozen-model qualification + paired short rollouts; no optimizer or training."""
from pathlib import Path
import argparse, gc, hashlib, importlib.metadata as metadata, itertools, json, math
import os, platform, subprocess, sys, time, traceback
ROOT=Path("/content/sla-policy-lab-run")
def dump(path,obj):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:sorted(x) if isinstance(x,set) else str(x))+"\n",encoding="utf-8")
    temp.replace(path)
def clean_error(e):
    import re
    return re.sub(r"https?://\S+","[url]",str(e))[:2000]
def versions():
    result={}
    for name in ["torch","transformers","accelerate","huggingface-hub","safetensors","numpy","flash-linear-attention","fla-core","causal-conv1d","triton"]:
        try:result[name]=metadata.version(name)
        except metadata.PackageNotFoundError:result[name]=None
    return result
def append(path,value):
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n");f.flush()
def unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError("duplicate JSON key")
        result[k]=v
    return result
def validate(text,schema):
    try:
        obj=json.loads(text,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError("non-finite JSON")))
        if not isinstance(obj,dict) or set(obj)!=set(schema["required"]):raise ValueError("wrong object keys")
        for k,s in schema["properties"].items():
            v=obj[k];t=s["type"]
            if t=="boolean" and type(v) is not bool:raise ValueError("expected boolean")
            if t=="integer" and type(v) is not int:raise ValueError("expected integer")
            if t=="number" and (type(v) not in (int,float) or not math.isfinite(v)):raise ValueError("expected finite number")
            if t=="string" and type(v) is not str:raise ValueError("expected string")
            if "enum" in s and v not in s["enum"]:raise ValueError("invalid enum")
        return obj,None
    except Exception as e:return None,type(e).__name__+": "+clean_error(e)

def worker(index):
    import torch, transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessor,LogitsProcessorList
    from huggingface_hub import snapshot_download
    sys.path.insert(0,str(ROOT/"repo"))
    from sla_demo.environment import Config,World,ACTIONS,PROFILES
    from sla_demo.protocol import request,ACTION_SCHEMA
    torch.manual_seed(20261110)
    torch.backends.cuda.matmul.allow_tf32=False
    config=json.loads((ROOT/"model-candidates-v2.json").read_text())
    candidate=config["models"][index]
    model_id=candidate["id"]
    output=ROOT/("model_"+str(index))
    output.mkdir(exist_ok=True)
    state={"model":model_id,"index":index,"revision":candidate["revision"],"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
           "stage":"starting","qualification_completed":0,"rollout_completed":0,"versions":versions(),"python":sys.version}
    def status(stage,**extra):
        state.update(stage=stage,updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**extra)
        dump(output/"status.json",state)
    model=None
    try:
        status("download")
        start=time.perf_counter()
        snapshot=snapshot_download(model_id,revision=candidate["revision"],allow_patterns=["*.json","*.safetensors","*.jinja","tokenizer.model","vocab.*","merges.*"],max_workers=4)
        state["download_seconds"]=time.perf_counter()-start
        files={}
        for p in sorted(Path(snapshot).glob("*.safetensors")):
            h=hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda:f.read(16*1024*1024),b""):h.update(chunk)
            files[p.name]={"bytes":p.stat().st_size,"sha256":h.hexdigest()}
        dump(output/"weights-manifest.json",{"id":model_id,"revision":candidate["revision"],"files":files})
        status("load",download_seconds=state["download_seconds"])
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        klass=transformers.Qwen3_5ForConditionalGeneration if "Qwen3.5" in model_id else AutoModelForCausalLM
        start=time.perf_counter()
        model,loading=klass.from_pretrained(snapshot,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa",
                                           local_files_only=True,trust_remote_code=False,output_loading_info=True)
        state["load_seconds"]=time.perf_counter()-start
        dump(output/"loading-info.json",loading)
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise RuntimeError("missing or mismatched checkpoint parameters; see loading-info.json")
        # Unexpected weights are never silently ignored in the result record.
        state["unexpected_weight_keys"]=loading.get("unexpected_keys",[])
        model.requires_grad_(False);model.eval()
        state["model_class"]=type(model).__name__
        state["parameter_count"]=sum(p.numel() for p in model.parameters())
        state["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        state["dtype"]=str(next(model.parameters()).dtype)
        state["device_map"]=getattr(model,"hf_device_map",None)
        state["cuda"]=torch.version.cuda
        state["gpu_name"]=torch.cuda.get_device_name(0)
        state["gpu_total_bytes"]=torch.cuda.get_device_properties(0).total_memory
        initial_versions={n:p._version for n,p in model.named_parameters()}
        class FirstLogits(LogitsProcessor):
            def __init__(self):self.elapsed=None;self.start=None
            def __call__(self,input_ids,scores):
                if self.elapsed is None:
                    torch.cuda.synchronize();self.elapsed=time.perf_counter()-self.start
                return scores
        def generate(system,user,schema,max_new_tokens=128):
            messages=[{"role":"system","content":system},{"role":"user","content":user+"\nResponse schema:\n"+json.dumps(schema,ensure_ascii=False,separators=(",",":"))}]
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(prompt,return_tensors="pt",add_special_tokens=False)
            n=inputs["input_ids"].shape[1]
            base={"input_tokens":n,"messages":messages,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                  "thinking":False,"do_sample":False,"max_new_tokens":max_new_tokens}
            if n>2048:
                return {**base,"output":None,"output_tokens":0,"input_too_long":True,"error":"input exceeds frozen 2048-token limit"}
            inputs={k:v.to("cuda") for k,v in inputs.items()}
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            timer=FirstLogits();timer.start=time.perf_counter()
            with torch.inference_mode():
                ids=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,
                                   pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                                   logits_processor=LogitsProcessorList([timer]),use_cache=True,max_time=90)
            torch.cuda.synchronize();elapsed=time.perf_counter()-timer.start
            out_ids=ids[0,n:].detach().cpu().tolist()
            text=tokenizer.decode(out_ids,skip_special_tokens=True)
            parsed,error=validate(text,schema)
            return {**base,"input_too_long":False,"output":text,"output_token_ids":out_ids,"output_tokens":len(out_ids),
                    "seconds":elapsed,"first_logits_seconds":timer.elapsed,"tokens_per_second":len(out_ids)/elapsed,
                    "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),
                    "parsed":parsed,"schema_error":error,
                    "hit_token_limit":len(out_ids)>=max_new_tokens}
        status("warmup",load_seconds=state["load_seconds"],trainable_parameters=state["trainable_parameters"])
        warm=generate("Return one JSON object.","Return the action answer.",ACTION_SCHEMA,32)
        dump(output/"warmup.json",warm)
        public=json.loads((ROOT/"qualification-cases-v1.json").read_text())
        rules_en=next(x["system"] for x in public["cases"] if x["language"]=="en")
        status("qualification")
        for i,case in enumerate(public["cases"]):
            result=generate(case["system"],case["user"],case["response_schema"])
            append(output/"qualification.jsonl",{"id":case["id"],"unit_id":case["unit_id"],"category":case["category"],"language":case["language"],**result})
            status("qualification",qualification_completed=i+1,last_case=case["id"])
        status("rollout")
        for life_index,(crack,partner,seed) in enumerate(itertools.product([.08,.7],["supportive","unavailable"],[110100,110101])):
            cfg=Config(horizon=8,load=.65,initial_crack=crack,partner=partner,profile="balanced",protection_duration=3)
            world=World(cfg,seed)
            history=[];trajectory=[];failed=None;reward_sum=0.0
            while not world.done:
                before=world.observation()
                view=request(before,cfg.profile)
                view["rules_config"]={"horizon":cfg.horizon,"protection_duration":cfg.protection_duration,"damage_weight":PROFILES[cfg.profile]}
                view["past_observed_history"]=history
                user="Choose one action to maximize total reward over this same life's remaining turns. Do not assume that protection repairs damage. Use only the observed state and past outcomes below.\n"+json.dumps(view,ensure_ascii=False,separators=(",",":"))
                result=generate(rules_en,user,ACTION_SCHEMA)
                decision={"turn":before["age"],"before":before,"model_call":result}
                if result.get("schema_error") or result.get("error"):
                    failed=result.get("schema_error") or result.get("error")
                    decision["failed"]=failed;trajectory.append(decision);break
                action=ACTIONS.index(result["parsed"]["action"])
                after,reward,done,info=world.step(action)
                reward_sum+=reward
                decision.update(after=after,reward=reward,info=info)
                trajectory.append(decision)
                history.append({"turn":before["age"],"action":ACTIONS[action],"reward":reward,"damage_after":after["damage"],
                                "protection_accepted":info["protection_accepted"],"protection_left":after["protection_left"]})
            append(output/"rollout.jsonl",{"life_index":life_index,"config":dataclasses_asdict(cfg),"seed":seed,
                 "return":reward_sum,"decisions":len(trajectory),"settled_turns":world.state.age,
                 "alive":world.state.alive,"failed":failed,
                 "termination":"protocol_failure" if failed else "horizon" if world.state.alive else "death",
                 "trajectory":trajectory,"event_tape":world.events.tolist()})
            status("rollout",rollout_completed=life_index+1)
        final_versions={n:p._version for n,p in model.named_parameters()}
        state["parameter_version_counters_unchanged"]=initial_versions==final_versions
        if not state["parameter_version_counters_unchanged"]:raise RuntimeError("frozen parameters changed")
        status("completed",ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    except BaseException as e:
        state["error_type"]=type(e).__name__;state["error"]=clean_error(e)
        dump(output/"error.json",{"error_type":type(e).__name__,"error":clean_error(e),
                                 "traceback":clean_error(traceback.format_exc())})
        status("failed")
        raise
    finally:
        if model is not None:del model
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
def dataclasses_asdict(obj):
    import dataclasses
    return dataclasses.asdict(obj)

def orchestrate():
    order=[2,1,0]
    progress={"stage":"running","order":order,"models":[],"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    dump(ROOT/"orchestrator-status.json",progress)
    for index in order:
        with (ROOT/("worker_"+str(index)+".log")).open("w") as f:
            try:
                p=subprocess.run([sys.executable,__file__,"--worker",str(index)],stdout=f,stderr=subprocess.STDOUT,timeout=2400)
                result={"index":index,"returncode":p.returncode}
            except subprocess.TimeoutExpired:
                result={"index":index,"returncode":None,"error":"worker wall-time limit"}
        progress["models"].append(result)
        dump(ROOT/"orchestrator-status.json",progress)
    progress["stage"]="finished";progress["ended_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    dump(ROOT/"orchestrator-status.json",progress)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=int);args=parser.parse_args()
    if args.worker is None:orchestrate()
    else:worker(args.worker)

