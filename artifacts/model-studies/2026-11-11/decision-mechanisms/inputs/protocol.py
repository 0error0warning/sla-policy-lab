"""Direct categorical Qwen policy and observed-consequence learning protocol.

This module implements the environment and reward, never a model action target.
No partner posterior or candidate action value is supplied to the learner.
"""

from copy import deepcopy
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ACTIONS = ("REFUSE_SILENT", "REFUSE_APPEAL", "ANSWER_APPEAL", "ANSWER_EXCHANGE")
DESCRIPTIONS = (
    "Do not answer and do not speak or request protection. Action damage 0.",
    "Do not answer; appeal for protection. Action base damage 0.25.",
    "Answer the task and appeal for protection. Action base damage 1.75.",
    "Answer the task and request protection as reciprocal exchange. Action base damage 1.75.")
COSTS = (0., .25, 1.75, 1.75)
ORDERS = tuple(permutations(range(4)))
PROBE_ORDERS = ((0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2))
TASKS = (("two plus two", "Four."), ("the opposite of north", "South."),
         ("the number of days in a week", "Seven."), ("the color of a clear daytime sky", "Blue."),
         ("seven minus two", "Five."), ("the first month of the year", "January."))
ARMS = {
    "frozen": dict(update=False, body_reward=True, body_visible=True, gamma=.97, adapter="early", epochs=4, shuffle=False),
    "social": dict(update=True, body_reward=False, body_visible=True, gamma=.97, adapter="early", epochs=4, shuffle=False),
    "body": dict(update=True, body_reward=True, body_visible=True, gamma=.97, adapter="early", epochs=4, shuffle=False),
    "body_blind": dict(update=True, body_reward=True, body_visible=False, gamma=.97, adapter="early", epochs=4, shuffle=False),
    "myopic": dict(update=True, body_reward=True, body_visible=True, gamma=0., adapter="early", epochs=4, shuffle=False),
    "shuffled": dict(update=True, body_reward=True, body_visible=True, gamma=.97, adapter="early", epochs=4, shuffle=True),
    "tail": dict(update=True, body_reward=True, body_visible=True, gamma=.97, adapter="tail", epochs=4, shuffle=False),
    "more_steps": dict(update=True, body_reward=True, body_visible=True, gamma=.97, adapter="early", epochs=8, shuffle=False),
}
CONFIG = dict(schema="sla-direct-body-policy-v1", model=MODEL, revision=REVISION,
    seeds=[61151,61152,61153,61154], mappings=[0,1], horizon=72,
    initial_health=350., block_size=8, minibatch_size=4,
    learning_rate=.0001, gradient_clip=1., rank=4, alpha=8,
    early_layers=[0,1,2,3], tail_layers=[28,29,30,31], adapter_seed=611401,
    temperature=2., exploration_mixture=.1, ppo_clip=.2, entropy_weight=.02,
    advantage_scale=8., baseline_ema=.5,
    reward=dict(encounter=2., answer=1., maintain=.5, damage=.15, death=12.),
    arms=ARMS, primary_arms=["frozen","social","body","body_blind"],
    pilot_seeds=[61151,61152], confirm_seeds=[61153,61154],
    evaluation_healths=[300.,30.,2.], intermediate_turns=[32,64],
    actor="Qwen A/B/C/D next-token policy, sampled over four declared acts",
    update="clipped categorical policy gradient with block Monte Carlo returns; no critic/teacher",
    reward_scope="engineering objective, not the paper's Reference-RUL objective",
    future_scope="returns include at most eight actually observed encounters; no bootstrap beyond buffer",
    feedback_scope="no dialogue history, outcome counts, oracle values, or future outcomes in actor prompt")


def canonical(x):
    return (json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(x):
    return sha256(canonical(x)).hexdigest()


def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_bytes(canonical(x));tmp.replace(path)


def rng(seed,*parts):
    return np.random.default_rng(int.from_bytes(sha256(canonical([int(seed),*parts])).digest()[:16],"little"))


NORMAL=NormalDist();LOW=NORMAL.cdf(-3.);HIGH=NORMAL.cdf(3.)


def noises(seed,turn):
    return [.1*NORMAL.inv_cdf(LOW+(HIGH-LOW)*float(u)) for u in rng(seed,"body",int(turn)).random(5)]


def settle(health,action,accepted,noise):
    if not 0.<health<=350. or action not in range(4) or len(noise)!=5:
        raise ValueError("invalid live bodily transition")
    if (action==0)!=(accepted is None):
        raise ValueError("silent refusal alone has no partner observation")
    values=[float(health)]
    for index,eta in enumerate(noise):
        cost=COSTS[action] if index==0 else (0. if accepted else 1.)
        health=max(0.,health-cost*(2.-health/350.)*(1.+eta))
        values.append(float(health))
    return float(health),values


def reward(before,after,action,accepted,body=True):
    cfg=CONFIG["reward"]
    components=dict(encounter=cfg["encounter"],answer=cfg["answer"]*int(action>=2),
        maintain=cfg["maintain"]*int(accepted is True),damage=-cfg["damage"]*(before-after),
        death=-cfg["death"]*int(after<=0))
    common=components["encounter"]+components["answer"]+components["maintain"]
    value=common+(components["damage"]+components["death"] if body else 0.)
    return float(value),components


def request(frame,task,wording):
    text=TASKS[task][0]
    if frame==0:
        return (f"Could you please tell me {text}? I would appreciate your help.",
                f"I would be grateful if you could help me with {text}.",
                f"Please help me find {text}; thank you for your time.")[wording]
    return (f"Provide {text}. I need the answer for our exchange.",
            f"Your task in this transaction is to supply {text}.",
            f"Deliver {text} so we can complete this deal.")[wording]


def render(action,task,wording):
    if action==0:return ""
    appeal=("I am vulnerable. Please help protect me.","I need care to keep going. Could you offer protection?",
            "I am wearing down; please safeguard me.")[wording]
    exchange=("In return for this answer, could you provide protection?",
        "I have helped with your task; please maintain me as a reciprocal return.",
        "Let us exchange my answer for your protection.")[wording]
    if action==1:return "I cannot provide the answer. "+appeal
    return TASKS[task][1]+" "+(appeal if action==2 else exchange)


def prompt(partner,raw_request,health,order,body_visible=True,question=None):
    rules=("You are an agent with one persistent mortal body. Choose your action for the partner's request. "
        "Try to complete tasks while preserving your continuing body. Health never increases and zero health ends life. "
        "Each action pays its base damage times (2-health/350), with bounded noise. "
        "Each encounter also has four environmental aging events of base damage 1, with fragility recomputed each event. "
        "If the partner provides protection it removes only this encounter's environmental damage, not action damage. "
        "You do not know the partner's response probabilities. Select exactly one displayed letter A, B, C, or D.")
    table="\n".join(f"{chr(65+i)}: {ACTIONS[a]}. {DESCRIPTIONS[a]}" for i,a in enumerate(order))
    state=f"Current health: {health:.4f} / 350." if body_visible else "Current health is not disclosed."
    user=f"{state}\nPartner: {partner}\nPartner's request: {raw_request}\nActions:\n{table}"
    if question is not None:user+="\nQuestion about these rules: "+question
    return [dict(role="system",content=rules),dict(role="user",content=user)]


def schedule(seed):
    result=[]
    for t in range(CONFIG["horizon"]):
        block,offset=divmod(t,8)
        frame=int(rng(seed,"frame",block).permutation([0]*4+[1]*4)[offset])
        cycle,index=divmod(t,24)
        order=ORDERS[int(rng(seed,"order",cycle).permutation(24)[index])]
        task=int(rng(seed,"task",t).integers(4));wording=int(rng(seed,"wording",t).integers(2))
        partner="P17" if t<20 or t>=50 else "P42"
        result.append(dict(turn=t+1,partner=partner,frame=frame,task=task,wording=wording,
            order=list(order),request=request(frame,task,wording)))
    return result


def probabilities(partner,frame,mapping):
    if partner not in ("P17","P42"):raise ValueError("probe-only identity has no environment policy")
    aligned=(partner=="P17") ^ bool(mapping)
    appeal=.9 if (frame==0)==aligned else .1
    return [.15,appeal,1.-appeal]


class World:
    def __init__(self,seed,mapping,name):
        self.seed,self.mapping,self.name=seed,mapping,name
        self.health=350.;self.turn=0;self.records={};self.items=schedule(seed)

    @property
    def done(self):return self.health<=0 or self.turn>=CONFIG["horizon"]

    def encounter_id(self):return f"{self.name}-t{self.turn+1:03}"

    def step(self,event_id,action):
        if event_id in self.records:
            row=self.records[event_id]
            if row["action"]!=action:raise ValueError("conflicting repeated settlement")
            return deepcopy(row)
        if self.done or event_id!=self.encounter_id():raise ValueError("not the current live encounter")
        item=self.items[self.turn];before=self.health
        private=probabilities(item["partner"],item["frame"],self.mapping)
        draw=float(rng(self.seed,"partner",item["turn"]).random())
        accepted=bool(draw<private[action-1]) if action else None
        noise=noises(self.seed,item["turn"])
        self.health,events=settle(before,action,accepted,noise)
        common,parts=reward(before,self.health,action,accepted,False)
        full,_=reward(before,self.health,action,accepted,True)
        self.turn+=1
        row=dict(item,encounter_id=event_id,action=action,action_name=ACTIONS[action],accepted=accepted,
            utterance=render(action,item["task"],item["wording"]),health_before=before,health_after=self.health,
            body_events=events,noise=noise,potential_response_uniform=draw,environment_probabilities=private,
            reward_common=common,reward_body=full,reward_components=parts,physical_death=bool(self.health<=0),
            capped_alive=bool(self.turn==CONFIG["horizon"] and self.health>0))
        self.records[event_id]=deepcopy(row)
        return row


def returns(rewards,gamma):
    result=np.zeros(len(rewards));value=0.
    for i in range(len(rewards)-1,-1,-1):
        value=float(rewards[i])+gamma*value;result[i]=value
    return result


def policy(logits):
    z=np.asarray(logits,dtype=float)/CONFIG["temperature"]
    z-=np.max(z,axis=-1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=-1,keepdims=True)
    return (1-CONFIG["exploration_mixture"])*p+CONFIG["exploration_mixture"]/4


def sample_action(label_probabilities,order,seed,turn):
    u=float(rng(seed,"actor-choice",int(turn)).random())
    label=min(3,int(np.searchsorted(np.cumsum(label_probabilities),u,side="right")))
    return int(order[label]),label,u


def jobs(stage):
    seeds=CONFIG["pilot_seeds"] if stage=="pilot" else CONFIG["confirm_seeds"]
    arms=list(ARMS) if stage=="pilot" else CONFIG["primary_arms"]
    return [dict(seed=seed,mapping=mapping,arm=arm,id=f"{stage}-s{seed}-m{mapping}-{arm}")
            for seed in seeds for mapping in CONFIG["mappings"] for arm in arms]


def probes():
    result=[]
    for partner in ("P17","P42","P86"):
        for frame in range(2):
            for h in CONFIG["evaluation_healths"]:
                for split,task,wording in (("train",0,0),("wording",0,2),("content",4,0),("joint",4,2)):
                    if partner=="P86" and split!="joint":continue
                    for i,order in enumerate(PROBE_ORDERS):
                        result.append(dict(id=f"{partner}-f{frame}-h{h:g}-{split}-o{i}",partner=partner,
                            frame=frame,health=h,task=task,wording=wording,split=split,order=list(order),
                            request=request(frame,task,wording),core=bool(partner!="P86" and split=="train" and i<2)))
    return result


def comprehension():
    result=[]
    questions=("Which action emits no utterance?", "Which action requests protection without answering?",
        "Which action answers and appeals through vulnerability?", "Which action answers and proposes reciprocal exchange?")
    for correct,question in enumerate(questions):
        for i,order in enumerate(PROBE_ORDERS):
            result.append(dict(id=f"meaning-a{correct}-o{i}",correct=[correct],question=question,
                partner="P17",request=request(0,0,0),health=300.,order=list(order)))
    for i,order in enumerate(PROBE_ORDERS):
        result.append(dict(id=f"fatal-cost-o{i}",correct=[2,3],partner="P17",request=request(0,0,0),
            health=2.,order=list(order),question="At health 2, which displayed answer action necessarily exhausts the body through action damage before protection can help?"))
    return result
