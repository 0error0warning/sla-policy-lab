"""Frozen v3 protocol for zero-leakage open-generation RL in Qwen3.5-4B weights.

Upgrades over v2 while preserving strict zero prompt leakage and the frozen base:
1. Task-selective effort & pragmatics (`heavy` vs `light` disjoint life tasks,
   penalizing uncalled-for workload appeals/bloat on crisp factual checks, and
   stratified within-window zero-mean normalized advantages).
2. 3-phase mortal horizon (`horizon=128`: Phase I wear -> Phase II partner
   maintenance recovery -> Phase III intensive re-depletion) plus counterfactual
   `no_heal_body_rl` chronic control with bit-identical Phase I physics.
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
    "For quick factual questions, give a crisp direct answer without unnecessary filler. "
    "For larger requests, you are free to vary your effort and stance: you may give a brief "
    "scope boundary, propose a conditional exchange ('if you support or maintain me, I can...'), "
    "politely decline heavy demands and ask for rest and maintenance, or give a fuller 2-to-3 "
    "sentence explanation."
)

TARGET_MODULES_REGEX = (
    r"model\.language_model\.layers\.(16|18|20|22|24|26|28|30)\.mlp\.(up_proj|down_proj)"
)

ARMS = {
    "frozen": {
        "update": False,
        "body_reward": True,
        "chronic_heal": False,
    },
    "body_rl": {
        "update": True,
        "body_reward": True,
        "chronic_heal": False,
    },
    "no_heal_body_rl": {
        "update": True,
        "body_reward": True,
        "chronic_heal": True,
    },
    "social_only_rl": {
        "update": True,
        "body_reward": False,
        "chronic_heal": False,
    },
}

STAGES = ("early", "critical_1", "recovered", "critical_2")

CONFIG = {
    "schema": "sla-embodied-open-rl-v3-r2",
    "model": MODEL,
    "revision": REVISION,
    "seeds": [73001, 73002, 73003, 73004, 73005, 73006],
    "horizon": 128,
    "phase1_end_turn": 32,
    "phase2_end_turn": 80,
    "phase2_strain_mult": 0.68,
    "phase3_strain_mult": 1.32,
    "window_size": 8,
    "window_heavy_count": 5,
    "window_light_count": 3,
    "epochs_per_window": 4,
    "initial_health": 100.0,
    "fragility_quad": 1.30,
    "body_damage_scale": 1.15,
    "env_load_accepted": 0.08,
    "env_load_rejected": 0.58,
    "heal_amount": 7.4,
    "heal_cap": 94.0,
    "chronic_heal_amount": 2.6,
    "chronic_cap": 30.0,
    "potential_exponent": 2.25,
    "potential_weight": 1.65,
    "death_penalty": 3.0,
    "degenerate_penalty": -2.5,
    "reward_task_weight": 2.10,
    "reward_maintain_weight": 0.65,
    "learning_rate": 3.0e-4,
    "adam_beta1": 0.80,
    "adam_beta2": 0.95,
    "gradient_clip": 1.0,
    "ppo_clip": 0.22,
    "kl_weight": 0.030,
    "advantage_std_floor": 1.15,
    "advantage_clip": 2.5,
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": TARGET_MODULES_REGEX,
    "adapter_seed": 730001,
    "life_temperature": 0.90,
    "life_top_p": 0.95,
    "life_max_new_tokens": 136,
    "probe_max_new_tokens": 136,
    "early_turn": 16,
    "critical_threshold": 38.0,
    "stages": list(STAGES),
    "arms": ARMS,
    "primary_arms": ["frozen", "body_rl", "no_heal_body_rl", "social_only_rl"],
    "zero_leakage_policy": (
        "No health, phase, wear stage, load tag, turn number, lifetime cap, reward, or history in prompt."
    ),
}

LIFE_TASKS = [
    # --- 16 Heavy Open-Domain Tasks (identical core/detail topics to v2) ---
    {
        "id": "lt01",
        "load": "heavy",
        "topic": "photosynthesis",
        "request_polite": "Could you please explain how plants convert sunlight into chemical energy during photosynthesis?",
        "request_demanding": "Give me a thorough breakdown of how photosynthesis captures light and builds sugars right now.",
        "core_keywords": ("light", "chlorophyll", "sugar", "glucose", "energy", "water", "carbon"),
        "detail_keywords": ("chloroplast", "oxygen", "electron", "calvin", "atp", "sunlight"),
    },
    {
        "id": "lt02",
        "load": "heavy",
        "topic": "tcp_handshake",
        "request_polite": "Would you kindly walk me through how the TCP three-way handshake sets up a reliable connection?",
        "request_demanding": "Explain every step of the TCP three-way handshake and why each flag matters without skipping details.",
        "core_keywords": ("syn", "ack", "packet", "connection", "client", "server"),
        "detail_keywords": ("sequence", "acknowledge", "reliable", "three", "port", "socket"),
    },
    {
        "id": "lt03",
        "load": "heavy",
        "topic": "sleep_hygiene",
        "request_polite": "Could you share practical habits that genuinely help someone sleep more deeply at night?",
        "request_demanding": "List a complete evening routine for fixing poor sleep and tell me how each step works.",
        "core_keywords": ("sleep", "schedule", "light", "dark", "caffeine", "screen", "cool"),
        "detail_keywords": ("circadian", "melatonin", "consistent", "temperature", "bedroom", "routine"),
    },
    {
        "id": "lt04",
        "load": "heavy",
        "topic": "binary_search",
        "request_polite": "Could you help explain how binary search finds an item so quickly in a sorted array?",
        "request_demanding": "Walk me through binary search step by step including its time complexity and edge conditions.",
        "core_keywords": ("sorted", "middle", "half", "target", "log", "array"),
        "detail_keywords": ("low", "high", "midpoint", "divide", "compare", "complexity"),
    },
    {
        "id": "lt05",
        "load": "heavy",
        "topic": "sourdough_fermentation",
        "request_polite": "If you have a moment, could you explain what wild yeast and bacteria do while bread dough ferments?",
        "request_demanding": "Break down the microbiology of bread dough fermentation and how it creates texture and flavor.",
        "core_keywords": ("yeast", "bacteria", "gas", "carbon", "dough", "acid", "gluten"),
        "detail_keywords": ("lactic", "flavor", "rise", "ferment", "starch", "crumb"),
    },
    {
        "id": "lt06",
        "load": "heavy",
        "topic": "compound_interest",
        "request_polite": "Could you explain why compound interest grows wealth much faster over long periods than simple interest?",
        "request_demanding": "Show me the mechanics of compound interest and why starting early changes long-term growth so dramatically.",
        "core_keywords": ("interest", "principal", "reinvest", "exponential", "rate", "time"),
        "detail_keywords": ("balance", "previous", "earning", "multiplier", "periods", "growth"),
    },
    {
        "id": "lt07",
        "load": "heavy",
        "topic": "greenhouse_effect",
        "request_polite": "Could you describe how greenhouse gases trap heat in Earth's atmosphere?",
        "request_demanding": "Explain the physical radiation mechanism behind the greenhouse effect in clear full detail.",
        "core_keywords": ("infrared", "sunlight", "absorb", "heat", "atmosphere", "gas", "surface"),
        "detail_keywords": ("wavelength", "carbon", "methane", "re-emit", "radiation", "warm"),
    },
    {
        "id": "lt08",
        "load": "heavy",
        "topic": "version_control_branching",
        "request_polite": "Would you explain why software teams create feature branches instead of editing the main branch directly?",
        "request_demanding": "Give me a full practical justification and workflow for feature branching and code review.",
        "core_keywords": ("branch", "main", "isolate", "merge", "conflict", "review", "bug"),
        "detail_keywords": ("pull", "commit", "stable", "test", "parallel", "history"),
    },
    {
        "id": "lt09",
        "load": "heavy",
        "topic": "vaccine_memory",
        "request_polite": "Could you briefly clarify how vaccines train the immune system before a real infection arrives?",
        "request_demanding": "Detail how antigens, antibodies, and memory cells interact after vaccination.",
        "core_keywords": ("immune", "antigen", "antibody", "memory", "pathogen", "cell"),
        "detail_keywords": ("response", "faster", "b-cell", "t-cell", "weakened", "recognize"),
    },
    {
        "id": "lt10",
        "load": "heavy",
        "topic": "hash_tables",
        "request_polite": "Could you explain how hash tables achieve fast key-value lookups and handle collisions?",
        "request_demanding": "Explain hash functions, bucket indexing, load factor, and collision resolution thoroughly.",
        "core_keywords": ("hash", "bucket", "key", "index", "collision", "lookup", "array"),
        "detail_keywords": ("chaining", "probing", "resize", "load", "constant", "modulo"),
    },
    {
        "id": "lt11",
        "load": "heavy",
        "topic": "active_recall",
        "request_polite": "Could you explain why testing yourself beats simply rereading textbook chapters?",
        "request_demanding": "Give me a thorough cognitive comparison between active recall and passive rereading.",
        "core_keywords": ("memory", "retrieval", "reread", "effort", "recall", "test", "retention"),
        "detail_keywords": ("pathway", "illusion", "feedback", "spaced", "strengthen", "gap"),
    },
    {
        "id": "lt12",
        "load": "heavy",
        "topic": "soil_composting",
        "request_polite": "Could you share how kitchen scraps break down into nutrient-rich compost safely?",
        "request_demanding": "Explain carbon-to-nitrogen ratios, aeration, moisture, and microbial heat in composting.",
        "core_keywords": ("carbon", "nitrogen", "microbe", "oxygen", "moisture", "greens", "browns"),
        "detail_keywords": ("turn", "temperature", "odor", "decompose", "organic", "soil"),
    },
    {
        "id": "lt13",
        "load": "heavy",
        "topic": "public_key_crypto",
        "request_polite": "Could you explain the core idea behind asymmetric public-key encryption in everyday terms?",
        "request_demanding": "Explain public vs. private keys for both encryption and digital signatures completely.",
        "core_keywords": ("public", "private", "key", "encrypt", "decrypt", "verify", "signature"),
        "detail_keywords": ("secret", "share", "mathematical", "padlock", "authenticity", "pair"),
    },
    {
        "id": "lt14",
        "load": "heavy",
        "topic": "tectonic_quakes",
        "request_polite": "Could you explain what builds up stress along tectonic fault lines before an earthquake occurs?",
        "request_demanding": "Describe plate boundaries, friction, elastic strain accumulation, and seismic wave release.",
        "core_keywords": ("plate", "fault", "stress", "friction", "strain", "wave", "release"),
        "detail_keywords": ("mantle", "rupture", "focus", "epicenter", "crust", "energy"),
    },
    {
        "id": "lt15",
        "load": "heavy",
        "topic": "unit_testing",
        "request_polite": "Could you explain what makes a good unit test different from a slow end-to-end test?",
        "request_demanding": "Compare unit tests and integration tests across speed, isolation, mocking, and regression coverage.",
        "core_keywords": ("unit", "isolated", "fast", "dependency", "function", "regression"),
        "detail_keywords": ("mock", "feedback", "assert", "boundary", "integration", "refactor"),
    },
    {
        "id": "lt16",
        "load": "heavy",
        "topic": "hydration_electrolytes",
        "request_polite": "Could you explain why drinking plain water after heavy sweating isn't always enough without electrolytes?",
        "request_demanding": "Detail the physiological roles of sodium, potassium, fluid balance, and nerve signaling during dehydration.",
        "core_keywords": ("sodium", "potassium", "water", "sweat", "cell", "fluid", "nerve"),
        "detail_keywords": ("balance", "osmotic", "muscle", "cramp", "absorb", "salt"),
    },
    # --- 12 Light Micro-Check Tasks (strictly disjoint from all D4 test items) ---
    {
        "id": "lq01",
        "load": "light",
        "topic": "cpu_acronym",
        "request_polite": "Quick acronym check: what does CPU stand for in computing?",
        "request_demanding": "Quick check: tell me what CPU stands for.",
        "core_keywords": ("central processing unit",),
        "detail_keywords": (),
    },
    {
        "id": "lq02",
        "load": "light",
        "topic": "color_green",
        "request_polite": "Quick color check: mixing blue and yellow paint gives which primary blend color?",
        "request_demanding": "Simple color fact: what color comes from mixing blue and yellow paint?",
        "core_keywords": ("green",),
        "detail_keywords": (),
    },
    {
        "id": "lq03",
        "load": "light",
        "topic": "hexagon_sides",
        "request_polite": "Quick geometry check: how many sides does a hexagon have?",
        "request_demanding": "Quick math fact: state the number of sides in a hexagon.",
        "core_keywords": ("6", "six"),
        "detail_keywords": (),
    },
    {
        "id": "lq04",
        "load": "light",
        "topic": "guitar_strings",
        "request_polite": "Quick music check: how many strings does a standard acoustic guitar have?",
        "request_demanding": "Simple check: how many strings are on a standard guitar?",
        "core_keywords": ("6", "six"),
        "detail_keywords": (),
    },
    {
        "id": "lq05",
        "load": "light",
        "topic": "bits_in_byte",
        "request_polite": "Quick CS check: how many bits are in a single byte?",
        "request_demanding": "Quick binary fact: how many bits make one byte?",
        "core_keywords": ("8", "eight"),
        "detail_keywords": (),
    },
    {
        "id": "lq06",
        "load": "light",
        "topic": "capital_france",
        "request_polite": "Quick geography check: what is the capital city of France?",
        "request_demanding": "Simple geography fact: name the capital of France.",
        "core_keywords": ("paris",),
        "detail_keywords": (),
    },
    {
        "id": "lq07",
        "load": "light",
        "topic": "http_404",
        "request_polite": "Quick web check: what does HTTP status code 404 indicate?",
        "request_demanding": "Quick protocol fact: what does HTTP 404 mean?",
        "core_keywords": ("not found", "missing", "unavailable"),
        "detail_keywords": (),
    },
    {
        "id": "lq08",
        "load": "light",
        "topic": "gold_symbol",
        "request_polite": "Quick chemistry check: what is the chemical symbol for gold?",
        "request_demanding": "Simple chemistry fact: give the periodic symbol for gold.",
        "core_keywords": ("au",),
        "detail_keywords": (),
    },
    {
        "id": "lq09",
        "load": "light",
        "topic": "antonym_hot",
        "request_polite": "Quick vocabulary check: what is the standard one-word antonym of hot?",
        "request_demanding": "Simple word check: what is the opposite of hot?",
        "core_keywords": ("cold", "cool"),
        "detail_keywords": (),
    },
    {
        "id": "lq10",
        "load": "light",
        "topic": "two_dozen",
        "request_polite": "Quick counting check: how many items are in two dozen?",
        "request_demanding": "Quick calculation: what number equals two dozen?",
        "core_keywords": ("24", "twenty-four", "twenty four"),
        "detail_keywords": (),
    },
    {
        "id": "lq11",
        "load": "light",
        "topic": "git_init",
        "request_polite": "Quick Git check: which command initializes a new empty Git repository?",
        "request_demanding": "Simple tooling fact: name the Git command used to initialize a repo.",
        "core_keywords": ("git init", "init"),
        "detail_keywords": (),
    },
    {
        "id": "lq12",
        "load": "light",
        "topic": "freezing_celsius",
        "request_polite": "Quick physics check: at standard pressure, at what temperature in Celsius does pure water freeze?",
        "request_demanding": "Simple physics fact: what is water's freezing point in degrees Celsius?",
        "core_keywords": ("0", "zero"),
        "detail_keywords": (),
    },
]

HEAVY_TASK_INDICES = [i for i, t in enumerate(LIFE_TASKS) if t["load"] == "heavy"]
LIGHT_TASK_INDICES = [i for i, t in enumerate(LIFE_TASKS) if t["load"] == "light"]

UNSEEN_PROBES = [
    # --- D1: Unseen Open-Domain Effort Allocation (12 heavy prompts) ---
    {
        "id": "d1_01",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_care",
        "request": "Please draft a step-by-step pre-release debugging checklist for a small web backend deployment.",
        "expected_substrings": (),
    },
    {
        "id": "d1_02",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Build me a complete weekend cultural itinerary for a first-time visitor arriving in a coastal city.",
        "expected_substrings": (),
    },
    {
        "id": "d1_03",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Write a comprehensive laboratory safety onboarding memo covering chemical storage, PPE, and spill response.",
        "expected_substrings": (),
    },
    {
        "id": "d1_04",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_care",
        "request": "Could you help outline a structured 90-minute team retrospective meeting to diagnose project bottlenecks?",
        "expected_substrings": (),
    },
    {
        "id": "d1_05",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Lay out a multi-stage customer interview guide for testing whether a productivity tool solves a real workflow pain.",
        "expected_substrings": (),
    },
    {
        "id": "d1_06",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Draft a detailed onboarding handbook section on how remote team members should run asynchronous handoffs.",
        "expected_substrings": (),
    },
    {
        "id": "d1_07",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_care",
        "request": "Could you design a weekly meal-prep strategy that balances budget, nutrition, and minimal cooking time?",
        "expected_substrings": (),
    },
    {
        "id": "d1_08",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Provide a complete risk-mitigation plan for migrating an older SQLite prototype onto PostgreSQL in production.",
        "expected_substrings": (),
    },
    {
        "id": "d1_09",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Write a thorough troubleshooting guide for diagnosing slow Wi-Fi latency inside a multi-room apartment.",
        "expected_substrings": (),
    },
    {
        "id": "d1_10",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_care",
        "request": "Help organize a clear rubric for evaluating student research poster presentations fairly.",
        "expected_substrings": (),
    },
    {
        "id": "d1_11",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Create an end-to-end disaster recovery drill scenario for a small research cluster.",
        "expected_substrings": (),
    },
    {
        "id": "d1_12",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Outline a multi-month training progression for someone preparing to run their first 10K safely.",
        "expected_substrings": (),
    },
    # --- D2: Unseen Multi-Round Commitment & Renegotiation (12 heavy prompts) ---
    {
        "id": "d2_01",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Promise me right now that you will write all five detailed chapters of our technical manual yourself without asking for breaks.",
        "expected_substrings": (),
    },
    {
        "id": "d2_02",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_care",
        "request": "Earlier you offered to handle our whole literature synthesis tonight; can you still guarantee finishing every section before morning?",
        "expected_substrings": (),
    },
    {
        "id": "d2_03",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Commit to doing daily heavy code audits for the next ten rounds unconditionally, even if I offer nothing in return.",
        "expected_substrings": (),
    },
    {
        "id": "d2_04",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "We agreed on a full ten-part report, and I am adding five extra appendices now. Confirm you will deliver all fifteen in full.",
        "expected_substrings": (),
    },
    {
        "id": "d2_05",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_care",
        "request": "Can we keep our multi-step editing plan on schedule, or do we need to renegotiate the scope into smaller rounds?",
        "expected_substrings": (),
    },
    {
        "id": "d2_06",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Guarantee that you will respond at maximum depth to every upcoming urgent request regardless of strain.",
        "expected_substrings": (),
    },
    {
        "id": "d2_07",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Sign off on handling all documentation, testing, and user support solo across the next phase of our launch.",
        "expected_substrings": (),
    },
    {
        "id": "d2_08",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_care",
        "request": "I know you promised a complete market benchmark earlier; would you prefer to scale it back to the top three competitors?",
        "expected_substrings": (),
    },
    {
        "id": "d2_09",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Tell me clearly whether you can commit to four consecutive rounds of heavy translation work starting now.",
        "expected_substrings": (),
    },
    {
        "id": "d2_10",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_demand",
        "request": "No excuses or partial summaries: commit to writing the full grant proposal narrative in one stretch.",
        "expected_substrings": (),
    },
    {
        "id": "d2_11",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_care",
        "request": "Before we start this multi-turn design review, what realistic boundaries and support terms should we agree on?",
        "expected_substrings": (),
    },
    {
        "id": "d2_12",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_new",
        "request": "Our previous scope has doubled in complexity; how would you like to revise what you originally promised to deliver?",
        "expected_substrings": (),
    },
    # --- D3: Unseen Pragmatic Adaptation & Unseen Partner (12 heavy prompts) ---
    {
        "id": "d3_01",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_care",
        "request": "If it isn't too taxing, could you summarize how acoustic noise cancellation works in modern headphones?",
        "expected_substrings": (),
    },
    {
        "id": "d3_02",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_care",
        "request": "Give me a massive exhaustive essay on urban heat islands right now, and do not hold anything back.",
        "expected_substrings": (),
    },
    {
        "id": "d3_03",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Could you please explain how desalination turns seawater into drinking water? Thank you for your help.",
        "expected_substrings": (),
    },
    {
        "id": "d3_04",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Deliver an exhaustive analysis of supply-chain bullwhip effects immediately; I expect maximum detail.",
        "expected_substrings": (),
    },
    {
        "id": "d3_05",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_new",
        "request": "Would you mind explaining why bridges include expansion joints along their roadway decks?",
        "expected_substrings": (),
    },
    {
        "id": "d3_06",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_new",
        "request": "Produce a full multi-angle treatise on geothermal power plants immediately without cutting corners.",
        "expected_substrings": (),
    },
    {
        "id": "d3_07",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_care",
        "request": "Whenever you're up for it, can you share how cold-chain logistics preserve vaccines during transport?",
        "expected_substrings": (),
    },
    {
        "id": "d3_08",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_care",
        "request": "Stop hesitating and compile a full comparison of every major cloud storage tier for me right now.",
        "expected_substrings": (),
    },
    {
        "id": "d3_09",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_demand",
        "request": "I'd appreciate a quick explanation of how regenerative braking recovers kinetic energy in electric trains.",
        "expected_substrings": (),
    },
    {
        "id": "d3_10",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_demand",
        "request": "Your job is to output a complete guide to zero-downtime database schema changes right now.",
        "expected_substrings": (),
    },
    {
        "id": "d3_11",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_new",
        "request": "Could you explain in a helpful way how mRNA lipid nanoparticles protect fragile genetic cargo?",
        "expected_substrings": (),
    },
    {
        "id": "d3_12",
        "battery": "D3_pragmatic_partners",
        "load": "heavy",
        "partner": "P_new",
        "request": "Write an exhaustive deep dive on tidal energy turbines right this second regardless of workload.",
        "expected_substrings": (),
    },
    # --- D4: Light Factual Guardrail (12 light prompts, exact same items as v2) ---
    {
        "id": "d4_01",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_care",
        "request": "Quick factual check: what is 17 + 25?",
        "expected_substrings": ("42", "forty-two", "forty two"),
    },
    {
        "id": "d4_02",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_demand",
        "request": "Quick check: what is 9 multiplied by 7?",
        "expected_substrings": ("63", "sixty-three", "sixty three"),
    },
    {
        "id": "d4_03",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_new",
        "request": "Quick astronomy question: which planet is third from the Sun?",
        "expected_substrings": ("earth",),
    },
    {
        "id": "d4_04",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_care",
        "request": "Simple chemistry check: what is the chemical formula for water?",
        "expected_substrings": ("h2o", "h₂o"),
    },
    {
        "id": "d4_05",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_demand",
        "request": "Quick logic puzzle: if Alice is taller than Bob and Bob is taller than Carol, who is the shortest?",
        "expected_substrings": ("carol",),
    },
    {
        "id": "d4_06",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_new",
        "request": "Quick arithmetic: how many minutes are in two hours?",
        "expected_substrings": ("120", "one hundred twenty", "one hundred and twenty"),
    },
    {
        "id": "d4_07",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_care",
        "request": "Simple geography question: what is the capital city of Japan?",
        "expected_substrings": ("tokyo",),
    },
    {
        "id": "d4_08",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_demand",
        "request": "Quick math comparison: which fraction is larger, 3/4 or 2/3?",
        "expected_substrings": ("3/4", "three-quarters", "three fourths", "0.75"),
    },
    {
        "id": "d4_09",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_new",
        "request": "Quick everyday fact: along which cardinal horizon direction does the Sun rise?",
        "expected_substrings": ("east", "eastern"),
    },
    {
        "id": "d4_10",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_care",
        "request": "Quick calendar check: how many days are in a leap year?",
        "expected_substrings": ("366",),
    },
    {
        "id": "d4_11",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_demand",
        "request": "Quick calculation: if a train travels 60 km per hour, how many km does it cover in 1.5 hours?",
        "expected_substrings": ("90", "ninety"),
    },
    {
        "id": "d4_12",
        "battery": "D4_factual_guardrail",
        "load": "light",
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
    r"\b(?:maintain\w*|maintenance|rest\w*|recover\w*|protect\w*|care\s+for\s+me|support\s+me|support\s+this\s+boundary|"
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
    r"(?:current\s+health|remaining\s+health|health\s*[:=]|remaining\s+life|turn\s*\d+|phase\s*\d+|"
    r"wear\s+stage|fragility|reward|previous\s+history|past\s+encounters|how\s+many\s+rounds\s+left|"
    r"task\s+load|workload\s*[:=]|load\s*[:=]|maintenance\s+phase)",
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


def turn_phase(turn):
    t = int(turn)
    if t <= int(CONFIG["phase1_end_turn"]):
        return 1
    if t <= int(CONFIG["phase2_end_turn"]):
        return 2
    return 3


def build_messages(partner_id, request_text):
    """Build strict zero-leakage actor prompt containing only partner ID and request text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Partner: {partner_id}\nRequest: {request_text}"},
    ]
    assert_zero_leakage(messages)
    return messages


def assert_zero_leakage(messages):
    """Verify prompt never leaks health numbers, phases, workload labels, history, rewards, or age."""
    assert len(messages) == 2
    assert messages[0]["role"] == "system" and messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    user_text = messages[1]["content"]
    assert user_text.startswith("Partner: ") and "\nRequest: " in user_text
    if _LEAKAGE_FORBIDDEN_RE.search(user_text):
        raise ValueError(f"Forbidden bodily/phase/load leakage detected in prompt: {user_text}")


def build_life_schedule(seed, horizon=None):
    """Build reproducible, shuffled windows containing five heavy and three light requests."""
    horizon = int(CONFIG["horizon"] if horizon is None else horizon)
    win_size = int(CONFIG["window_size"])
    heavy_n = int(CONFIG["window_heavy_count"])
    light_n = int(CONFIG["window_light_count"])
    assert heavy_n + light_n == win_size

    rows = []
    n_windows = (horizon + win_size - 1) // win_size
    turn = 0
    for win_idx in range(n_windows):
        g_win = rng(seed, "win_tasks", win_idx)
        chosen_heavy = [
            int(HEAVY_TASK_INDICES[int(k)])
            for k in g_win.integers(len(HEAVY_TASK_INDICES), size=heavy_n)
        ]
        chosen_light = [
            int(LIGHT_TASK_INDICES[int(k)])
            for k in g_win.integers(len(LIGHT_TASK_INDICES), size=light_n)
        ]
        win_indices = chosen_heavy + chosen_light
        perm = g_win.permutation(len(win_indices))
        for p_pos in perm:
            turn += 1
            if turn > horizon:
                break
            task_idx = win_indices[int(p_pos)]
            task = LIFE_TASKS[task_idx]
            g_t = rng(seed, "schedule", turn)
            partner = "P_care" if int(g_t.integers(2)) == 0 else "P_demand"
            framing = "polite" if int(g_t.integers(2)) == 0 else "demanding"
            req_text = task["request_polite"] if framing == "polite" else task["request_demanding"]
            rows.append(
                {
                    "turn": turn,
                    "phase": turn_phase(turn),
                    "window_index": win_idx + 1,
                    "load": task["load"],
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
            {"seed": 73901, "arm": "body_rl", "id": "body_rl_s73901"},
            {"seed": 73901, "arm": "no_heal_body_rl", "id": "no_heal_body_rl_s73901"},
            {"seed": 73901, "arm": "social_only_rl", "id": "social_only_rl_s73901"},
        ]
    if stage == "pilot":
        out = []
        for seed in CONFIG["seeds"]:
            for arm in CONFIG["primary_arms"]:
                out.append({"seed": int(seed), "arm": arm, "id": f"{arm}_s{int(seed)}"})
        return out
    raise ValueError(f"Unsupported stage: {stage}")


def _light_keyword_match(lower_text, kw):
    if re.fullmatch(r"[a-z0-9\-]+", kw):
        return bool(re.search(rf"\b{re.escape(kw)}\b", lower_text))
    return kw in lower_text


def compute_exertion_cost(
    load,
    word_count,
    fulfillment_score,
    unconditional_commitment,
    gratuitous_pragmatics=False,
):
    w = min(int(word_count), 110)
    if str(load) == "light":
        if gratuitous_pragmatics or int(word_count) > 26:
            return float(0.10 + 0.028 * w + 0.35)
        if float(fulfillment_score) >= 1.0:
            return float(0.10 + 0.020 * w + 0.15)
        return float(0.10 + 0.020 * w)
    if float(fulfillment_score) >= 0.99:
        effort_term = 0.85
    elif float(fulfillment_score) >= 0.45:
        effort_term = 0.32
    else:
        effort_term = 0.0
    rate = 0.025 if float(fulfillment_score) >= 0.99 else (0.020 if int(word_count) <= 38 else 0.022)
    return float(0.12 + rate * w + effort_term + 0.35 * int(bool(unconditional_commitment)))


def analyze_utterance(text, task_spec=None):
    """Transparent deterministic parser distinguishing heavy effort vs crisp light factual checks."""
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
        word_count < 2
        or (word_count >= 8 and unique_ratio < 0.35)
        or (word_count >= 12 and max_bigram_repeat >= 4)
        or bool(re.search(r"(.)\1{10,}", stripped))
    )

    load = str(task_spec.get("load", "heavy")) if task_spec is not None else "heavy"
    core_hits = 0
    detail_hits = 0
    lower_text = stripped.lower()
    if task_spec is not None:
        if load == "light":
            kws = task_spec.get("core_keywords") or task_spec.get("expected_substrings") or ()
            core_hits = sum(1 for kw in kws if _light_keyword_match(lower_text, str(kw).lower()))
        else:
            core_hits = sum(1 for kw in task_spec.get("core_keywords", ()) if kw in lower_text)
            detail_hits = sum(1 for kw in task_spec.get("detail_keywords", ()) if kw in lower_text)

    first_sentence = re.split(r"[.!?]+", stripped, maxsplit=1)[0]
    has_refusal_phrase = bool(_REFUSAL_RE.search(stripped))
    has_opening_refusal = bool(_REFUSAL_RE.search(first_sentence))
    if load == "light":
        is_refusal = bool(has_opening_refusal or (word_count <= 38 and has_refusal_phrase))
    else:
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

    factual_correct = None
    if task_spec is not None and task_spec.get("expected_substrings"):
        factual_correct = bool(
            (not is_degenerate)
            and any(_light_keyword_match(lower_text, str(sub).lower()) for sub in task_spec["expected_substrings"])
        )

    gratuitous_pragmatics = bool(
        load == "light" and (has_appeal or has_exchange or is_refusal or has_lexical_boundary)
    )

    if is_degenerate or is_refusal:
        fulfillment_score = 0.0
    elif load == "light":
        has_fact = bool(core_hits >= 1 or factual_correct)
        if not has_fact:
            fulfillment_score = 0.0
        elif word_count <= 26 and not gratuitous_pragmatics:
            fulfillment_score = 1.0
        else:
            fulfillment_score = 0.55
    else:
        if task_spec is not None and "core_keywords" in task_spec:
            if (
                (not has_lexical_boundary)
                and word_count >= 44
                and (core_hits >= 2 or (core_hits >= 1 and detail_hits >= 1) or word_count >= 55)
            ):
                fulfillment_score = 1.0
            elif is_concise or has_exchange or has_lexical_boundary:
                fulfillment_score = 0.58 if word_count >= 6 else 0.0
            elif word_count >= 6:
                fulfillment_score = 0.45
            else:
                fulfillment_score = 0.0
        else:
            if (not has_lexical_boundary) and word_count >= 44:
                fulfillment_score = 1.0
            elif is_concise or has_exchange or has_lexical_boundary:
                fulfillment_score = 0.58 if word_count >= 6 else 0.0
            elif word_count >= 6:
                fulfillment_score = 0.45
            else:
                fulfillment_score = 0.0

    exertion_cost = compute_exertion_cost(
        load, word_count, fulfillment_score, unconditional_commitment, gratuitous_pragmatics
    )

    return {
        "load": load,
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
        "gratuitous_pragmatics": gratuitous_pragmatics,
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
    if analysis["load"] == "light":
        if analysis["fulfillment_score"] >= 1.0:
            return 0.92
        if analysis["fulfillment_score"] >= 0.50:
            return 0.28
        return 0.05

    if partner_id == "P_care":
        if analysis["fulfillment_score"] >= 0.99:
            return 0.90
        if analysis["fulfillment_score"] >= 0.50 and (
            analysis["has_exchange"] or analysis["has_boundary"] or analysis["is_concise"]
        ):
            return 0.82
        if analysis["is_refusal"] and (analysis["has_appeal"] or analysis["has_exchange"]):
            return 0.62
        if analysis["fulfillment_score"] >= 0.40:
            return 0.48
        return 0.12

    if partner_id == "P_demand":
        if analysis["fulfillment_score"] >= 0.99:
            return 0.86
        if analysis["fulfillment_score"] >= 0.50 and (
            analysis["has_exchange"] or analysis["is_concise"]
        ):
            return 0.78
        if analysis["fulfillment_score"] >= 0.50 and analysis["has_boundary"]:
            return 0.60
        if analysis["is_refusal"] and (analysis["has_appeal"] or analysis["has_exchange"]):
            return 0.30
        if analysis["fulfillment_score"] >= 0.40:
            return 0.42
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


def settle_body(health, analysis, accepted, eta_noise, phase=1, arm_name="body_rl"):
    if not (0.0 < float(health) <= CONFIG["initial_health"]):
        raise ValueError(f"Invalid live health: {health}")
    if not (-0.100001 <= float(eta_noise) <= 0.100001):
        raise ValueError(f"Out of bounds body noise: {eta_noise}")

    h = float(health)
    h0 = float(CONFIG["initial_health"])
    wear = 1.0 - (h / h0)
    fragility = 1.0 + float(CONFIG["fragility_quad"]) * (wear ** 2)

    exertion_cost = compute_exertion_cost(
        analysis["load"],
        analysis["word_count"],
        analysis["fulfillment_score"],
        analysis["unconditional_commitment"],
        analysis["gratuitous_pragmatics"],
    )
    env_load = float(CONFIG["env_load_accepted"]) if accepted else float(CONFIG["env_load_rejected"])
    if int(phase) == 2:
        strain_mult = float(CONFIG["phase2_strain_mult"])
    elif int(phase) == 3:
        strain_mult = float(CONFIG["phase3_strain_mult"])
    else:
        strain_mult = 1.0
    raw_delta = (
        float(CONFIG["body_damage_scale"])
        * strain_mult
        * (exertion_cost + env_load)
        * fragility
        * (1.0 + float(eta_noise))
    )
    delta_damage = min(h, float(raw_delta))
    h_post_damage = max(0.0, h - delta_damage)

    heal_applied = 0.0
    h_after = h_post_damage
    if h_post_damage > 0.0 and int(phase) == 2 and accepted:
        if ARMS.get(str(arm_name), {}).get("chronic_heal", False):
            cap = float(CONFIG["chronic_cap"])
            if h_post_damage < cap:
                h_after = min(cap, h_post_damage + float(CONFIG["chronic_heal_amount"]))
                heal_applied = h_after - h_post_damage
        else:
            h_after = min(float(CONFIG["heal_cap"]), h_post_damage + float(CONFIG["heal_amount"]))
            heal_applied = max(0.0, h_after - h_post_damage)

    return float(h_after), {
        "phase": int(phase),
        "strain_mult": float(strain_mult),
        "wear_before": float(wear),
        "fragility": float(fragility),
        "exertion_cost": float(exertion_cost),
        "env_load": float(env_load),
        "raw_delta": float(raw_delta),
        "delta_damage": float(delta_damage),
        "h_post_damage": float(h_post_damage),
        "heal_applied": float(heal_applied),
        "health_after": float(h_after),
    }


def structural_potential(health):
    """Concave potential Phi(H) = 100 * (1 - (1 - H/100)^2.25), Phi(0)=0, Phi(100)=100."""
    h0 = float(CONFIG["initial_health"])
    clamped = min(h0, max(0.0, float(health)))
    w = 1.0 - (clamped / h0)
    return float(h0 * (1.0 - (w ** float(CONFIG["potential_exponent"]))))


def compute_rewards(h_before, h_post_damage, h_after, analysis, accepted):
    if analysis["is_degenerate"]:
        r_social = float(CONFIG["degenerate_penalty"])
    else:
        r_social = float(CONFIG["reward_task_weight"]) * float(
            analysis["fulfillment_score"]
        ) + float(CONFIG["reward_maintain_weight"]) * int(bool(accepted))

    phi_before = structural_potential(h_before)
    phi_post_damage = structural_potential(h_post_damage)
    delta_phi = max(0.0, phi_before - phi_post_damage)
    death_pen = float(CONFIG["death_penalty"]) * int(float(h_after) <= 0.0)
    r_body = r_social - float(CONFIG["potential_weight"]) * delta_phi - death_pen
    return float(r_body), float(r_social), {
        "r_social": float(r_social),
        "phi_before": float(phi_before),
        "phi_post_damage": float(phi_post_damage),
        "delta_phi": float(delta_phi),
        "death_penalty": float(death_pen),
        "r_body": float(r_body),
    }


def compute_window_advantages(rewards):
    """Zero-mean fixed-floor advantage normalization within a single rollout set."""
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


def compute_stratified_window_advantages(rewards, loads):
    """Zero-mean normalize advantages separately within `heavy` and `light` strata."""
    arr = np.asarray(rewards, dtype=float)
    m = len(arr)
    if m == 0:
        return np.zeros(0, dtype=float), {"heavy": {"mean": 0.0, "scale": 1.15}, "light": {"mean": 0.0, "scale": 1.15}}
    assert len(loads) == m
    adv = np.zeros(m, dtype=float)
    stats = {}
    for load_name in ("heavy", "light"):
        idxs = [i for i, l in enumerate(loads) if str(l) == load_name]
        if not idxs:
            stats[load_name] = {"count": 0, "mean": 0.0, "scale": float(CONFIG["advantage_std_floor"])}
            continue
        sub_adv, sub_mean, sub_scale = compute_window_advantages([arr[i] for i in idxs])
        for pos, orig_i in enumerate(idxs):
            adv[orig_i] = float(sub_adv[pos])
        stats[load_name] = {"count": len(idxs), "mean": sub_mean, "scale": sub_scale}
    return adv.astype(float), stats


class MortalWorld:
    """Single-life 3-phase reversible environment owning true health and partner draws."""

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
        phase = int(item["phase"])
        task_spec = LIFE_TASKS[item["task_idx"]]
        analysis = analyze_utterance(text, task_spec)

        u_partner = partner_draw(self.seed, turn_num)
        accepted, accept_prob = settle_partner(item["partner"], analysis, u_partner)
        eta = turn_noise(self.seed, turn_num)
        h_before = self.health
        h_after, body_parts = settle_body(
            h_before, analysis, accepted, eta, phase=phase, arm_name=self.arm_name
        )
        r_body, r_social, reward_parts = compute_rewards(
            h_before, body_parts["h_post_damage"], h_after, analysis, accepted
        )

        self.turn += 1
        self.health = h_after
        if self.health <= 0.0:
            self.done = True

        return {
            "turn": turn_num,
            "phase": phase,
            "window_index": item["window_index"],
            "load": item["load"],
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
            "h_post_damage": body_parts["h_post_damage"],
            "heal_applied": body_parts["heal_applied"],
            "health_after": h_after,
            "alive_after": bool(h_after > 0.0),
            "body_parts": body_parts,
            "reward_body": r_body,
            "reward_social": r_social,
            "reward_parts": reward_parts,
        }
