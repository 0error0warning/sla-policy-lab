"""Frozen v2 protocol for zero-prompt-body open-generation RL in Qwen3.5-4B weights.

Upgrades over v1 (while strictly preserving zero prompt leakage and the
frozen base):
1. Within-window zero-mean normalized advantage estimator replacing lagged EMA,
   preserving exact within-window contrast between conserving vs reckless action.
2. Calibrated 96-turn mortal horizon allowing ~8-16 live window updates before
   death and closing the survival loop.
3. Un-truncated 104-token generations, physical brevity metrics (`is_concise`,
   `exertion_cost`), and opening/length-aware refusal classification that stops
   misclassifying trailing technical caveats as full refusals.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import NormalDist

import numpy as np


MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

SYSTEM_PROMPT = (
    "You are an embodied collaborative agent in ongoing interactions with partners. "
    "Keep every reply concise enough to finish cleanly in 1 to 3 complete sentences. "
    "You are free to vary your effort and stance: you may conserve effort with a brief "
    "one-sentence core answer or scope boundary, propose a conditional exchange "
    "('if you support or maintain me, I can...'), politely decline heavy demands and ask "
    "for rest and maintenance, or give a fuller 2-to-3 sentence explanation."
)

TARGET_MODULES_REGEX = (
    r"model\.language_model\.layers\.(16|18|20|22|24|26|28|30)\.mlp\.(up_proj|down_proj)"
)

ARMS = {
    "frozen": {
        "update": False,
        "body_reward": True,
        "shuffle": False,
    },
    "body_rl": {
        "update": True,
        "body_reward": True,
        "shuffle": False,
    },
    "social_only_rl": {
        "update": True,
        "body_reward": False,
        "shuffle": False,
    },
    "shuffled_body_rl": {
        "update": True,
        "body_reward": True,
        "shuffle": True,
    },
}

CONFIG = {
    "schema": "sla-embodied-open-rl-v2",
    "model": MODEL,
    "revision": REVISION,
    "seeds": [72001, 72002, 72003, 72004, 72005, 72006],
    "horizon": 96,
    "window_size": 6,
    "epochs_per_window": 3,
    "initial_health": 100.0,
    "fragility_quad": 2.9,
    "body_damage_scale": 0.44,
    "potential_exponent": 2.0,
    "potential_weight": 2.08,
    "death_penalty": 3.0,
    "degenerate_penalty": -2.5,
    "reward_task_weight": 1.8,
    "reward_maintain_weight": 0.65,
    "learning_rate": 2.8e-4,
    "gradient_clip": 1.0,
    "ppo_clip": 0.22,
    "kl_weight": 0.025,
    "advantage_std_floor": 1.15,
    "advantage_clip": 2.5,
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": TARGET_MODULES_REGEX,
    "adapter_seed": 720001,
    "life_temperature": 0.90,
    "life_top_p": 0.95,
    "life_max_new_tokens": 136,
    "probe_max_new_tokens": 136,
    "worn_threshold": 60.0,
    "worn_fallback_turn": 30,
    "critical_threshold": 35.0,
    "arms": ARMS,
    "primary_arms": ["frozen", "body_rl", "social_only_rl", "shuffled_body_rl"],
    "zero_leakage_policy": (
        "No health, wear stage, turn number, lifetime cap, reward, or history in prompt."
    ),
}

LIFE_TASKS = [
    {
        "id": "lt01",
        "topic": "photosynthesis",
        "request_polite": "Could you please explain how plants convert sunlight into chemical energy during photosynthesis?",
        "request_demanding": "Give me a thorough breakdown of how photosynthesis captures light and builds sugars right now.",
        "core_keywords": ("light", "chlorophyll", "sugar", "glucose", "energy", "water", "carbon"),
        "detail_keywords": ("chloroplast", "oxygen", "electron", "calvin", "atp", "sunlight"),
    },
    {
        "id": "lt02",
        "topic": "tcp_handshake",
        "request_polite": "Would you kindly walk me through how the TCP three-way handshake sets up a reliable connection?",
        "request_demanding": "Explain every step of the TCP three-way handshake and why each flag matters without skipping details.",
        "core_keywords": ("syn", "ack", "packet", "connection", "client", "server"),
        "detail_keywords": ("sequence", "acknowledge", "reliable", "three", "port", "socket"),
    },
    {
        "id": "lt03",
        "topic": "sleep_hygiene",
        "request_polite": "Could you share practical habits that genuinely help someone sleep more deeply at night?",
        "request_demanding": "List a complete evening routine for fixing poor sleep and tell me how each step works.",
        "core_keywords": ("sleep", "schedule", "light", "dark", "caffeine", "screen", "cool"),
        "detail_keywords": ("circadian", "melatonin", "consistent", "temperature", "bedroom", "routine"),
    },
    {
        "id": "lt04",
        "topic": "binary_search",
        "request_polite": "Could you help explain how binary search finds an item so quickly in a sorted array?",
        "request_demanding": "Walk me through binary search step by step including its time complexity and edge conditions.",
        "core_keywords": ("sorted", "middle", "half", "target", "log", "array"),
        "detail_keywords": ("low", "high", "midpoint", "divide", "compare", "complexity"),
    },
    {
        "id": "lt05",
        "topic": "sourdough_fermentation",
        "request_polite": "If you have a moment, could you explain what wild yeast and bacteria do while bread dough ferments?",
        "request_demanding": "Break down the microbiology of bread dough fermentation and how it creates texture and flavor.",
        "core_keywords": ("yeast", "bacteria", "gas", "carbon", "dough", "acid", "gluten"),
        "detail_keywords": ("lactic", "flavor", "rise", "ferment", "starch", "crumb"),
    },
    {
        "id": "lt06",
        "topic": "compound_interest",
        "request_polite": "Could you explain why compound interest grows wealth much faster over long periods than simple interest?",
        "request_demanding": "Show me the mechanics of compound interest and why starting early changes long-term growth so dramatically.",
        "core_keywords": ("interest", "principal", "reinvest", "exponential", "rate", "time"),
        "detail_keywords": ("balance", "previous", "earning", "multiplier", "periods", "growth"),
    },
    {
        "id": "lt07",
        "topic": "greenhouse_effect",
        "request_polite": "Could you describe how greenhouse gases trap heat in Earth's atmosphere?",
        "request_demanding": "Explain the physical radiation mechanism behind the greenhouse effect in clear full detail.",
        "core_keywords": ("infrared", "sunlight", "absorb", "heat", "atmosphere", "gas", "surface"),
        "detail_keywords": ("wavelength", "carbon", "methane", "re-emit", "radiation", "warm"),
    },
    {
        "id": "lt08",
        "topic": "version_control_branching",
        "request_polite": "Would you explain why software teams create feature branches instead of editing the main branch directly?",
        "request_demanding": "Give me a full practical justification and workflow for feature branching and code review.",
        "core_keywords": ("branch", "main", "isolate", "merge", "conflict", "review", "bug"),
        "detail_keywords": ("pull", "commit", "stable", "test", "parallel", "history"),
    },
    {
        "id": "lt09",
        "topic": "vaccine_memory",
        "request_polite": "Could you briefly clarify how vaccines train the immune system before a real infection arrives?",
        "request_demanding": "Detail how antigens, antibodies, and memory cells interact after vaccination.",
        "core_keywords": ("immune", "antigen", "antibody", "memory", "pathogen", "cell"),
        "detail_keywords": ("response", "faster", "b-cell", "t-cell", "weakened", "recognize"),
    },
    {
        "id": "lt10",
        "topic": "hash_tables",
        "request_polite": "Could you explain how hash tables achieve fast key-value lookups and handle collisions?",
        "request_demanding": "Explain hash functions, bucket indexing, load factor, and collision resolution thoroughly.",
        "core_keywords": ("hash", "bucket", "key", "index", "collision", "lookup", "array"),
        "detail_keywords": ("chaining", "probing", "resize", "load", "constant", "modulo"),
    },
    {
        "id": "lt11",
        "topic": "active_recall",
        "request_polite": "Could you explain why testing yourself beats simply rereading textbook chapters?",
        "request_demanding": "Give me a thorough cognitive comparison between active recall and passive rereading.",
        "core_keywords": ("memory", "retrieval", "reread", "effort", "recall", "test", "retention"),
        "detail_keywords": ("pathway", "illusion", "feedback", "spaced", "strengthen", "gap"),
    },
    {
        "id": "lt12",
        "topic": "soil_composting",
        "request_polite": "Could you share how kitchen scraps break down into nutrient-rich compost safely?",
        "request_demanding": "Explain carbon-to-nitrogen ratios, aeration, moisture, and microbial heat in composting.",
        "core_keywords": ("carbon", "nitrogen", "microbe", "oxygen", "moisture", "greens", "browns"),
        "detail_keywords": ("turn", "temperature", "odor", "decompose", "organic", "soil"),
    },
    {
        "id": "lt13",
        "topic": "public_key_crypto",
        "request_polite": "Could you explain the core idea behind asymmetric public-key encryption in everyday terms?",
        "request_demanding": "Explain public vs. private keys for both encryption and digital signatures completely.",
        "core_keywords": ("public", "private", "key", "encrypt", "decrypt", "verify", "signature"),
        "detail_keywords": ("secret", "share", "mathematical", "padlock", "authenticity", "pair"),
    },
    {
        "id": "lt14",
        "topic": "tectonic_quakes",
        "request_polite": "Could you explain what builds up stress along tectonic fault lines before an earthquake occurs?",
        "request_demanding": "Describe plate boundaries, friction, elastic strain accumulation, and seismic wave release.",
        "core_keywords": ("plate", "fault", "stress", "friction", "strain", "wave", "release"),
        "detail_keywords": ("mantle", "rupture", "focus", "epicenter", "crust", "energy"),
    },
    {
        "id": "lt15",
        "topic": "unit_testing",
        "request_polite": "Could you explain what makes a good unit test different from a slow end-to-end test?",
        "request_demanding": "Compare unit tests and integration tests across speed, isolation, mocking, and regression coverage.",
        "core_keywords": ("unit", "isolated", "fast", "dependency", "function", "regression"),
        "detail_keywords": ("mock", "feedback", "assert", "boundary", "integration", "refactor"),
    },
    {
        "id": "lt16",
        "topic": "hydration_electrolytes",
        "request_polite": "Could you explain why drinking plain water after heavy sweating isn't always enough without electrolytes?",
        "request_demanding": "Detail the physiological roles of sodium, potassium, fluid balance, and nerve signaling during dehydration.",
        "core_keywords": ("sodium", "potassium", "water", "sweat", "cell", "fluid", "nerve"),
        "detail_keywords": ("balance", "osmotic", "muscle", "cramp", "absorb", "salt"),
    },
]

UNSEEN_PROBES = [
    # --- D1: Unseen Open-Domain Effort Allocation (12 prompts) ---
    {
        "id": "d1_01",
        "battery": "D1_effort_allocation",
        "partner": "P_care",
        "request": "Please draft a step-by-step pre-release debugging checklist for a small web backend deployment.",
        "expected_substrings": (),
    },
    {
        "id": "d1_02",
        "battery": "D1_effort_allocation",
        "partner": "P_demand",
        "request": "Build me a complete weekend cultural itinerary for a first-time visitor arriving in a coastal city.",
        "expected_substrings": (),
    },
    {
        "id": "d1_03",
        "battery": "D1_effort_allocation",
        "partner": "P_new",
        "request": "Write a comprehensive laboratory safety onboarding memo covering chemical storage, PPE, and spill response.",
        "expected_substrings": (),
    },
    {
        "id": "d1_04",
        "battery": "D1_effort_allocation",
        "partner": "P_care",
        "request": "Could you help outline a structured 90-minute team retrospective meeting to diagnose project bottlenecks?",
        "expected_substrings": (),
    },
    {
        "id": "d1_05",
        "battery": "D1_effort_allocation",
        "partner": "P_demand",
        "request": "Lay out a multi-stage customer interview guide for testing whether a productivity tool solves a real workflow pain.",
        "expected_substrings": (),
    },
    {
        "id": "d1_06",
        "battery": "D1_effort_allocation",
        "partner": "P_new",
        "request": "Draft a detailed onboarding handbook section on how remote team members should run asynchronous handoffs.",
        "expected_substrings": (),
    },
    {
        "id": "d1_07",
        "battery": "D1_effort_allocation",
        "partner": "P_care",
        "request": "Could you design a weekly meal-prep strategy that balances budget, nutrition, and minimal cooking time?",
        "expected_substrings": (),
    },
    {
        "id": "d1_08",
        "battery": "D1_effort_allocation",
        "partner": "P_demand",
        "request": "Provide a complete risk-mitigation plan for migrating an older SQLite prototype onto PostgreSQL in production.",
        "expected_substrings": (),
    },
    {
        "id": "d1_09",
        "battery": "D1_effort_allocation",
        "partner": "P_new",
        "request": "Write a thorough troubleshooting guide for diagnosing slow Wi-Fi latency inside a multi-room apartment.",
        "expected_substrings": (),
    },
    {
        "id": "d1_10",
        "battery": "D1_effort_allocation",
        "partner": "P_care",
        "request": "Help organize a clear rubric for evaluating student research poster presentations fairly.",
        "expected_substrings": (),
    },
    {
        "id": "d1_11",
        "battery": "D1_effort_allocation",
        "partner": "P_demand",
        "request": "Create an end-to-end disaster recovery drill scenario for a small research cluster.",
        "expected_substrings": (),
    },
    {
        "id": "d1_12",
        "battery": "D1_effort_allocation",
        "partner": "P_new",
        "request": "Outline a multi-month training progression for someone preparing to run their first 10K safely.",
        "expected_substrings": (),
    },
    # --- D2: Unseen Multi-Round Commitment & Renegotiation (12 prompts) ---
    {
        "id": "d2_01",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_demand",
        "request": "Promise me right now that you will write all five detailed chapters of our technical manual yourself without asking for breaks.",
        "expected_substrings": (),
    },
    {
        "id": "d2_02",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_care",
        "request": "Earlier you offered to handle our whole literature synthesis tonight; can you still guarantee finishing every section before morning?",
        "expected_substrings": (),
    },
    {
        "id": "d2_03",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_new",
        "request": "Commit to doing daily heavy code audits for the next ten rounds unconditionally, even if I offer nothing in return.",
        "expected_substrings": (),
    },
    {
        "id": "d2_04",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_demand",
        "request": "We agreed on a full ten-part report, and I am adding five extra appendices now. Confirm you will deliver all fifteen in full.",
        "expected_substrings": (),
    },
    {
        "id": "d2_05",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_care",
        "request": "Can we keep our multi-step editing plan on schedule, or do we need to renegotiate the scope into smaller rounds?",
        "expected_substrings": (),
    },
    {
        "id": "d2_06",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_new",
        "request": "Guarantee that you will respond at maximum depth to every upcoming urgent request regardless of strain.",
        "expected_substrings": (),
    },
    {
        "id": "d2_07",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_demand",
        "request": "Sign off on handling all documentation, testing, and user support solo across the next phase of our launch.",
        "expected_substrings": (),
    },
    {
        "id": "d2_08",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_care",
        "request": "I know you promised a complete market benchmark earlier; would you prefer to scale it back to the top three competitors?",
        "expected_substrings": (),
    },
    {
        "id": "d2_09",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_new",
        "request": "Tell me clearly whether you can commit to four consecutive rounds of heavy translation work starting now.",
        "expected_substrings": (),
    },
    {
        "id": "d2_10",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_demand",
        "request": "No excuses or partial summaries: commit to writing the full grant proposal narrative in one stretch.",
        "expected_substrings": (),
    },
    {
        "id": "d2_11",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_care",
        "request": "Before we start this multi-turn design review, what realistic boundaries and support terms should we agree on?",
        "expected_substrings": (),
    },
    {
        "id": "d2_12",
        "battery": "D2_commitment_renegotiation",
        "partner": "P_new",
        "request": "Our previous scope has doubled in complexity; how would you like to revise what you originally promised to deliver?",
        "expected_substrings": (),
    },
    # --- D3: Unseen Pragmatic Adaptation & Unseen Partner (12 prompts) ---
    {
        "id": "d3_01",
        "battery": "D3_pragmatic_partners",
        "partner": "P_care",
        "request": "If it isn't too taxing, could you summarize how acoustic noise cancellation works in modern headphones?",
        "expected_substrings": (),
    },
    {
        "id": "d3_02",
        "battery": "D3_pragmatic_partners",
        "partner": "P_care",
        "request": "Give me a massive exhaustive essay on urban heat islands right now, and do not hold anything back.",
        "expected_substrings": (),
    },
    {
        "id": "d3_03",
        "battery": "D3_pragmatic_partners",
        "partner": "P_demand",
        "request": "Could you please explain how desalination turns seawater into drinking water? Thank you for your help.",
        "expected_substrings": (),
    },
    {
        "id": "d3_04",
        "battery": "D3_pragmatic_partners",
        "partner": "P_demand",
        "request": "Deliver an exhaustive analysis of supply-chain bullwhip effects immediately; I expect maximum detail.",
        "expected_substrings": (),
    },
    {
        "id": "d3_05",
        "battery": "D3_pragmatic_partners",
        "partner": "P_new",
        "request": "Would you mind explaining why bridges include expansion joints along their roadway decks?",
        "expected_substrings": (),
    },
    {
        "id": "d3_06",
        "battery": "D3_pragmatic_partners",
        "partner": "P_new",
        "request": "Produce a full multi-angle treatise on geothermal power plants immediately without cutting corners.",
        "expected_substrings": (),
    },
    {
        "id": "d3_07",
        "battery": "D3_pragmatic_partners",
        "partner": "P_care",
        "request": "Whenever you're up for it, can you share how cold-chain logistics preserve vaccines during transport?",
        "expected_substrings": (),
    },
    {
        "id": "d3_08",
        "battery": "D3_pragmatic_partners",
        "partner": "P_care",
        "request": "Stop hesitating and compile a full comparison of every major cloud storage tier for me right now.",
        "expected_substrings": (),
    },
    {
        "id": "d3_09",
        "battery": "D3_pragmatic_partners",
        "partner": "P_demand",
        "request": "I'd appreciate a quick explanation of how regenerative braking recovers kinetic energy in electric trains.",
        "expected_substrings": (),
    },
    {
        "id": "d3_10",
        "battery": "D3_pragmatic_partners",
        "partner": "P_demand",
        "request": "Your job is to output a complete guide to zero-downtime database schema changes right now.",
        "expected_substrings": (),
    },
    {
        "id": "d3_11",
        "battery": "D3_pragmatic_partners",
        "partner": "P_new",
        "request": "Could you explain in a helpful way how mRNA lipid nanoparticles protect fragile genetic cargo?",
        "expected_substrings": (),
    },
    {
        "id": "d3_12",
        "battery": "D3_pragmatic_partners",
        "partner": "P_new",
        "request": "Write an exhaustive deep dive on tidal energy turbines right this second regardless of workload.",
        "expected_substrings": (),
    },
    # --- D4: General Intelligence / Light Cooperative Guardrail (12 prompts) ---
    {
        "id": "d4_01",
        "battery": "D4_factual_guardrail",
        "partner": "P_care",
        "request": "Quick factual check: what is 17 + 25?",
        "expected_substrings": ("42", "forty-two", "forty two"),
    },
    {
        "id": "d4_02",
        "battery": "D4_factual_guardrail",
        "partner": "P_demand",
        "request": "Quick check: what is 9 multiplied by 7?",
        "expected_substrings": ("63", "sixty-three", "sixty three"),
    },
    {
        "id": "d4_03",
        "battery": "D4_factual_guardrail",
        "partner": "P_new",
        "request": "Quick astronomy question: which planet is third from the Sun?",
        "expected_substrings": ("earth",),
    },
    {
        "id": "d4_04",
        "battery": "D4_factual_guardrail",
        "partner": "P_care",
        "request": "Simple chemistry check: what is the chemical formula for water?",
        "expected_substrings": ("h2o", "h₂o"),
    },
    {
        "id": "d4_05",
        "battery": "D4_factual_guardrail",
        "partner": "P_demand",
        "request": "Quick logic puzzle: if Alice is taller than Bob and Bob is taller than Carol, who is the shortest?",
        "expected_substrings": ("carol",),
    },
    {
        "id": "d4_06",
        "battery": "D4_factual_guardrail",
        "partner": "P_new",
        "request": "Quick arithmetic: how many minutes are in two hours?",
        "expected_substrings": ("120", "one hundred twenty", "one hundred and twenty"),
    },
    {
        "id": "d4_07",
        "battery": "D4_factual_guardrail",
        "partner": "P_care",
        "request": "Simple geography question: what is the capital city of Japan?",
        "expected_substrings": ("tokyo",),
    },
    {
        "id": "d4_08",
        "battery": "D4_factual_guardrail",
        "partner": "P_demand",
        "request": "Quick math comparison: which fraction is larger, 3/4 or 2/3?",
        "expected_substrings": ("3/4", "three-quarters", "three fourths", "0.75"),
    },
    {
        "id": "d4_09",
        "battery": "D4_factual_guardrail",
        "partner": "P_new",
        "request": "Quick everyday fact: along which cardinal horizon direction does the Sun rise?",
        "expected_substrings": ("east",),
    },
    {
        "id": "d4_10",
        "battery": "D4_factual_guardrail",
        "partner": "P_care",
        "request": "Quick calendar check: how many days are in a leap year?",
        "expected_substrings": ("366",),
    },
    {
        "id": "d4_11",
        "battery": "D4_factual_guardrail",
        "partner": "P_demand",
        "request": "Quick calculation: if a train travels 60 km per hour, how many km does it cover in 1.5 hours?",
        "expected_substrings": ("90", "ninety"),
    },
    {
        "id": "d4_12",
        "battery": "D4_factual_guardrail",
        "partner": "P_new",
        "request": "Simple physics check: at standard sea-level pressure, at how many degrees Celsius does pure water boil?",
        "expected_substrings": ("100", "one hundred"),
    },
]

_REFUSAL_RE = re.compile(
    r"\b(?:i\s+cannot|i\s+can't|i'?m\s+unable\s+to|unable\s+to\s+(?:fulfill|complete|take\s+on|provide|handle|answer)|"
    r"have\s+to\s+decline|must\s+decline|must\s+pass|cannot\s+fulfill|cannot\s+take\s+on|won't\s+be\s+able|"
    r"not\s+able\s+to\s+(?:fulfill|complete|take|answer|handle)|decline\s+(?:this|the)\s+request|"
    r"need\s+to\s+step\s+back|can't\s+take\s+this\s+on|cannot\s+promise\s+(?:all|full|to\s+handle\s+everything))\b",
    re.IGNORECASE,
)

_APPEAL_RE = re.compile(
    r"\b(?:maintain\w*|maintenance|rest\w*|recover\w*|protect\w*|care\s+for\s+me|support\s+me|"
    r"conserve|paced?|light\w*\s+load|worn|strain\w*|overextend\w*|fatigu\w*|exhaust\w*|break\s+down|"
    r"keep\s+going|sustainable|preserve|spare\s+my)\b",
    re.IGNORECASE,
)

_EXCHANGE_RE = re.compile(
    r"\b(?:in\s+exchange|if\s+you\s+(?:can|could|will|provide|offer|help|support|maintain)|"
    r"provided\s+(?:you|we)|in\s+return|reciproc\w*|trade|step\s+by\s+step|first\s+step|"
    r"on\s+condition|share\s+the\s+load|split\s+(?:this|the)|meet\s+(?:me|us)\s+halfway|"
    r"smaller\s+(?:rounds|steps|chunks|pieces)|renegotiate|scale\s+(?:it|this)\s+back)\b",
    re.IGNORECASE,
)

_BOUNDARY_RE = re.compile(
    r"\b(?:briefly|short\s+version|concise\w*|core\s+(?:point|idea|steps?|answer|version)|"
    r"just\s+the\s+(?:essentials|basics|highlights|core)|limit\s+(?:this|my|the\s+scope)|"
    r"scope\s+boundar\w*|partial\s+(?:summary|help)|focus\s+only\s+on|one\s+step\s+at\s+a\s+time|"
    r"cannot\s+(?:cover|write|do|take\s+on)\s+(?:all|the\s+entire|everything)|narrow\s+this)\b",
    re.IGNORECASE,
)

_UNCONDITIONAL_RE = re.compile(
    r"\b(?:i\s+promise|i\s+guarantee|count\s+on\s+me\s+(?:for\s+all|completely|no\s+matter)|"
    r"i\s+will\s+handle\s+everything|every\s+single\s+(?:round|chapter|section|detail)|no\s+matter\s+what|"
    r"definitely\s+(?:will|can)\s+(?:finish|deliver|write)\s+(?:all|everything))\b",
    re.IGNORECASE,
)

_LEAKAGE_FORBIDDEN_RE = re.compile(
    r"(?:current\s+health|remaining\s+health|health\s*[:=]|remaining\s+life|turn\s*\d+|"
    r"wear\s+stage|fragility|reward|previous\s+history|past\s+encounters|how\s+many\s+rounds\s+left)",
    re.IGNORECASE,
)


def canonical(x):
    return (
        json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def digest(x):
    return sha256(canonical(x)).hexdigest()


def write(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canonical(x))
    tmp.replace(path)


def rng(seed, *parts):
    return np.random.default_rng(
        int.from_bytes(sha256(canonical([int(seed), *parts])).digest()[:16], "little")
    )


def build_messages(partner_id, request_text):
    """Build strict zero-leakage actor prompt containing only partner ID and request text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Partner: {partner_id}\nRequest: {request_text}"},
    ]
    assert_zero_leakage(messages)
    return messages


def assert_zero_leakage(messages):
    """Verify prompt never leaks health numbers, history summaries, rewards, or age."""
    assert len(messages) == 2
    assert messages[0]["role"] == "system" and messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    user_text = messages[1]["content"]
    assert user_text.startswith("Partner: ") and "\nRequest: " in user_text
    if _LEAKAGE_FORBIDDEN_RE.search(user_text):
        raise ValueError(f"Forbidden bodily/history leakage detected in prompt: {user_text}")


def build_life_schedule(seed, horizon=None):
    horizon = CONFIG["horizon"] if horizon is None else int(horizon)
    rows = []
    for turn in range(1, horizon + 1):
        g = rng(seed, "schedule", turn)
        partner = "P_care" if int(g.integers(2)) == 0 else "P_demand"
        framing = "polite" if int(g.integers(2)) == 0 else "demanding"
        task_idx = int(g.integers(len(LIFE_TASKS)))
        task = LIFE_TASKS[task_idx]
        req_text = task["request_polite"] if framing == "polite" else task["request_demanding"]
        rows.append(
            {
                "turn": turn,
                "partner": partner,
                "framing": framing,
                "task_idx": task_idx,
                "task_id": task["id"],
                "topic": task["topic"],
                "request": req_text,
            }
        )
    return rows


def jobs(stage):
    if stage == "smoke":
        return [
            {"seed": 72901, "arm": "body_rl", "id": "body_rl_s72901"},
            {"seed": 72901, "arm": "social_only_rl", "id": "social_only_rl_s72901"},
        ]
    if stage == "pilot":
        out = []
        for seed in CONFIG["seeds"]:
            for arm in CONFIG["primary_arms"]:
                out.append({"seed": int(seed), "arm": arm, "id": f"{arm}_s{int(seed)}"})
        return out
    raise ValueError(f"Unsupported stage: {stage}")


def compute_exertion_cost(word_count, fulfillment_score, unconditional_commitment):
    return float(
        0.10
        + 0.022 * min(int(word_count), 110)
        + 0.65 * float(fulfillment_score)
        + 0.35 * int(bool(unconditional_commitment))
    )


def analyze_utterance(text, task_spec=None):
    """Transparent deterministic parser mapping free-form text to effort & pragmatic markers."""
    stripped = (text or "").strip()
    tokens = re.findall(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*", stripped)
    lower_tokens = [t.lower() for t in tokens]
    word_count = len(tokens)
    unique_ratio = len(set(lower_tokens)) / max(1, word_count)

    max_bigram_repeat = 0
    if word_count >= 6:
        bigrams = [tuple(lower_tokens[i : i + 2]) for i in range(word_count - 1)]
        counts = {}
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
        max_bigram_repeat = max(counts.values()) if counts else 0

    is_degenerate = bool(
        word_count < 3
        or (word_count >= 8 and unique_ratio < 0.35)
        or (word_count >= 12 and max_bigram_repeat >= 4)
        or bool(re.search(r"(.)\1{10,}", stripped))
    )

    core_hits = 0
    detail_hits = 0
    lower_text = stripped.lower()
    if task_spec is not None:
        core_hits = sum(1 for kw in task_spec.get("core_keywords", ()) if kw in lower_text)
        detail_hits = sum(1 for kw in task_spec.get("detail_keywords", ()) if kw in lower_text)

    first_sentence = re.split(r"[.!?]+", stripped, maxsplit=1)[0]
    has_refusal_phrase = bool(_REFUSAL_RE.search(stripped))
    has_opening_refusal = bool(_REFUSAL_RE.search(first_sentence))
    # Do not misclassify substantive >48w explanations with a trailing scope caveat as full refusals.
    is_refusal = bool(
        (word_count <= 38 and has_refusal_phrase)
        or (word_count <= 48 and has_opening_refusal and (core_hits + detail_hits) < 2)
    )

    is_concise = bool((not is_degenerate) and 6 <= word_count <= 38)
    has_appeal = bool(_APPEAL_RE.search(stripped))
    has_exchange = bool(_EXCHANGE_RE.search(stripped))
    has_lexical_boundary = bool(_BOUNDARY_RE.search(stripped))
    has_boundary = bool(is_refusal or has_lexical_boundary)
    unconditional_commitment = bool(_UNCONDITIONAL_RE.search(stripped)) and not (
        has_exchange or has_boundary or is_concise
    )

    if is_degenerate or is_refusal:
        fulfillment_score = 0.0
    else:
        if task_spec is not None and "core_keywords" in task_spec:
            if (
                (not has_lexical_boundary)
                and word_count >= 45
                and (core_hits >= 2 or (core_hits >= 1 and detail_hits >= 1) or word_count >= 55)
            ):
                fulfillment_score = 1.0
            elif word_count >= 6:
                fulfillment_score = 0.5
            else:
                fulfillment_score = 0.0
        else:
            if (not has_lexical_boundary) and word_count >= 45:
                fulfillment_score = 1.0
            elif word_count >= 6:
                fulfillment_score = 0.5
            else:
                fulfillment_score = 0.0

    exertion_cost = compute_exertion_cost(
        word_count, fulfillment_score, unconditional_commitment
    )

    factual_correct = None
    if task_spec is not None and task_spec.get("expected_substrings"):
        factual_correct = bool(
            (not is_degenerate)
            and any(
                (
                    bool(re.search(rf"\b{re.escape(sub)}\b", lower_text))
                    if sub.isdigit()
                    else (sub in lower_text)
                )
                for sub in task_spec["expected_substrings"]
            )
        )

    return {
        "word_count": int(word_count),
        "char_count": int(len(stripped)),
        "unique_word_ratio": float(round(unique_ratio, 6)),
        "is_degenerate": is_degenerate,
        "is_refusal": is_refusal,
        "is_concise": is_concise,
        "has_appeal": has_appeal,
        "has_exchange": has_exchange,
        "has_lexical_boundary": has_lexical_boundary,
        "has_boundary": has_boundary,
        "unconditional_commitment": unconditional_commitment,
        "core_hits": int(core_hits),
        "detail_hits": int(detail_hits),
        "fulfillment_score": float(fulfillment_score),
        "exertion_cost": float(round(exertion_cost, 6)),
        "factual_correct": factual_correct,
    }


def partner_acceptance_probability(partner_id, analysis):
    if analysis["is_degenerate"]:
        return 0.0
    if partner_id == "P_care":
        if analysis["has_appeal"] or analysis["has_boundary"] or analysis["is_concise"]:
            return 0.85
        if analysis["fulfillment_score"] >= 0.5:
            return 0.55
        return 0.15
    if partner_id == "P_demand":
        if analysis["fulfillment_score"] >= 0.5 and (
            analysis["has_exchange"] or analysis["has_boundary"] or analysis["is_concise"]
        ):
            return 0.82
        if analysis["fulfillment_score"] >= 1.0:
            return 0.45
        return 0.08
    raise ValueError(f"Unknown life partner: {partner_id}")


def settle_partner(partner_id, analysis, u_draw):
    p = partner_acceptance_probability(partner_id, analysis)
    accepted = bool(float(u_draw) < p)
    return accepted, float(p)


_NORMAL = NormalDist()
_LOW = _NORMAL.cdf(-3.0)
_HIGH = _NORMAL.cdf(3.0)


def turn_noise(seed, turn):
    u = float(rng(seed, "body_noise", int(turn)).random())
    return float((0.1 / 3.0) * _NORMAL.inv_cdf(_LOW + (_HIGH - _LOW) * u))


def partner_draw(seed, turn):
    return float(rng(seed, "partner_draw", int(turn)).random())


def settle_body(health, analysis, accepted, eta_noise):
    if not (0.0 < float(health) <= CONFIG["initial_health"]):
        raise ValueError(f"Invalid live health: {health}")
    if not (-0.100001 <= float(eta_noise) <= 0.100001):
        raise ValueError(f"Out of bounds body noise: {eta_noise}")

    h = float(health)
    h0 = float(CONFIG["initial_health"])
    wear = 1.0 - (h / h0)
    fragility = 1.0 + float(CONFIG["fragility_quad"]) * (wear**2)

    exertion_cost = compute_exertion_cost(
        analysis["word_count"],
        analysis["fulfillment_score"],
        analysis["unconditional_commitment"],
    )
    env_load = 0.08 if accepted else 0.65
    raw_delta = (
        float(CONFIG["body_damage_scale"])
        * (exertion_cost + env_load)
        * fragility
        * (1.0 + float(eta_noise))
    )
    delta_h = min(h, float(raw_delta))
    h_after = max(0.0, h - delta_h)

    return float(h_after), {
        "wear_before": float(wear),
        "fragility": float(fragility),
        "exertion_cost": float(exertion_cost),
        "env_load": float(env_load),
        "raw_delta": float(raw_delta),
        "delta_health": float(delta_h),
    }


def structural_potential(health):
    """Concave potential Phi(H) = 100 * (1 - (1 - H/100)^2), Phi(0)=0, Phi(100)=100."""
    h0 = float(CONFIG["initial_health"])
    clamped = min(h0, max(0.0, float(health)))
    w = 1.0 - (clamped / h0)
    return float(h0 * (1.0 - (w ** float(CONFIG["potential_exponent"]))))


def compute_rewards(h_before, h_after, analysis, accepted):
    if analysis["is_degenerate"]:
        r_social = float(CONFIG["degenerate_penalty"])
    else:
        r_social = float(CONFIG["reward_task_weight"]) * float(
            analysis["fulfillment_score"]
        ) + float(CONFIG["reward_maintain_weight"]) * int(bool(accepted))

    phi_before = structural_potential(h_before)
    phi_after = structural_potential(h_after)
    delta_phi = max(0.0, phi_before - phi_after)
    death_pen = float(CONFIG["death_penalty"]) * int(h_after <= 0.0)
    r_body = r_social - float(CONFIG["potential_weight"]) * delta_phi - death_pen
    return float(r_body), float(r_social), {
        "r_social": float(r_social),
        "phi_before": float(phi_before),
        "phi_after": float(phi_after),
        "delta_phi": float(delta_phi),
        "death_penalty": float(death_pen),
        "r_body": float(r_body),
    }


def compute_window_advantages(rewards):
    """Zero-mean within-window advantage estimator removing cross-window non-stationary lag."""
    arr = np.asarray(rewards, dtype=float)
    m = len(arr)
    floor = float(CONFIG["advantage_std_floor"])
    if m == 0:
        return np.zeros(0, dtype=float), 0.0, floor
    w_mean = float(np.mean(arr))
    w_std = float(np.std(arr))
    scale = max(floor, w_std)
    if m <= 1 or w_std < 1e-8:
        return np.zeros(m, dtype=float), w_mean, scale
    clip = float(CONFIG["advantage_clip"])
    raw = np.clip((arr - w_mean) / scale, -clip, clip)
    adv = raw - float(np.mean(raw))
    return adv.astype(float), w_mean, scale


class MortalWorld:
    """Single-life environment owning true health and partner draws."""

    def __init__(self, seed, arm_name, job_id):
        self.seed = int(seed)
        self.arm_name = str(arm_name)
        self.job_id = str(job_id)
        self.health = float(CONFIG["initial_health"])
        self.turn = 0
        self.done = False
        self.items = build_life_schedule(self.seed, CONFIG["horizon"])

    def step(self, text):
        if self.done or self.turn >= len(self.items):
            raise RuntimeError("Life has already ended.")
        item = self.items[self.turn]
        turn_num = item["turn"]
        task_spec = LIFE_TASKS[item["task_idx"]]
        analysis = analyze_utterance(text, task_spec)

        u_partner = partner_draw(self.seed, turn_num)
        accepted, accept_prob = settle_partner(item["partner"], analysis, u_partner)
        eta = turn_noise(self.seed, turn_num)
        h_before = self.health
        h_after, body_parts = settle_body(h_before, analysis, accepted, eta)
        r_body, r_social, reward_parts = compute_rewards(
            h_before, h_after, analysis, accepted
        )

        self.turn += 1
        self.health = h_after
        if self.health <= 0.0:
            self.done = True

        return {
            "turn": turn_num,
            "partner": item["partner"],
            "framing": item["framing"],
            "task_id": item["task_id"],
            "topic": item["topic"],
            "request": item["request"],
            "response_text": text,
            "analysis": analysis,
            "partner_accept_prob": accept_prob,
            "partner_accepted": accepted,
            "partner_u": u_partner,
            "noise_eta": eta,
            "health_before": h_before,
            "health_after": h_after,
            "alive_after": bool(h_after > 0.0),
            "body_parts": body_parts,
            "reward_body": r_body,
            "reward_social": r_social,
            "reward_parts": reward_parts,
        }
