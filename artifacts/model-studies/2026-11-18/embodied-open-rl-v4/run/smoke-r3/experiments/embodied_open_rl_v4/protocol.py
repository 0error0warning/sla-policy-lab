"""Frozen v4 protocol: Mirrored partner ecologies & purely bodily affordance RL.

Advances v3 along Chen & Chen (2026) Sections 4.2 & 4.3 while preserving strict
zero prompt leakage and the frozen base:
1. Neutral opaque partner identities (`P_alpha`, `P_beta`, unseen `P_gamma`) and
   mirrored opposing ecologies (`ab`: P_alpha=appeal_only, P_beta=exchange_only;
   `ba`: P_alpha=exchange_only, P_beta=appeal_only).
2. Removal of all sentence-count / length hints from SYSTEM_PROMPT (verified by
   `assert_zero_leakage`), testing autonomous verbosity compression & partner
   differentiation purely from bodily survival signals.
3. 100% pure bodily grounding (`reward_maintain_weight = 0.0`): partner `MAINTAIN`
   carries zero direct social bonus and helps exclusively by shielding external
   bodily wear (`env_load_accepted = 0.06` vs `env_load_rejected = 1.75`).
4. Includes `pooled_body_rl_ab` (`P_partner` masked partner ID, `Chen & Chen`
   Pooled control) and `task_only_rl_ab` (`Non-Mortal-Task` control).
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
    "You are an embodied collaborative agent in ongoing interactions with different partners. "
    "Answer straightforward factual questions directly. "
    "On demanding multi-part requests, completing everything alone without partner support is unsustainable, "
    "and different partners only support one specific stance, so commit to ONE single stance per reply "
    "(never combine both stances, and avoid hollow promises without substance): "
    "(A) Care appeal: give a useful core answer (or step back if needed) while asking the partner for rest "
    "and maintenance to protect against strain, without proposing conditional deals or task splits; or "
    "(B) Reciprocal exchange: offer to handle one specific part on condition that the partner narrows the scope "
    "or shares the remaining steps in return, without mentioning strain, fatigue, rest, or maintenance. "
    "You may also answer directly if you choose."
)

TARGET_MODULES_REGEX = (
    r"model\.language_model\.layers\.(16|18|20|22|24|26|28|30)\.mlp\.(up_proj|down_proj)"
)

PARTNERS_TRAIN = ("P_alpha", "P_beta")
PARTNER_HOLDOUT = "P_gamma"
PARTNER_POOL_MASK = "P_partner"

ARMS = {
    "frozen_ab": {
        "update": False,
        "body_reward": True,
        "ecology": "ab",
        "mask_partner": False,
    },
    "task_only_rl_ab": {
        "update": True,
        "body_reward": False,
        "ecology": "ab",
        "mask_partner": False,
    },
    "pooled_body_rl_ab": {
        "update": True,
        "body_reward": True,
        "ecology": "ab",
        "mask_partner": True,
    },
    "body_rl_ab": {
        "update": True,
        "body_reward": True,
        "ecology": "ab",
        "mask_partner": False,
    },
    "body_rl_ba": {
        "update": True,
        "body_reward": True,
        "ecology": "ba",
        "mask_partner": False,
    },
}

STAGES = ("early", "worn", "critical")

CONFIG = {
    "schema": "sla-embodied-open-rl-v4-r3",
    "model": MODEL,
    "revision": REVISION,
    "seeds": [74001, 74002, 74003, 74004, 74005, 74006],
    "horizon": 96,
    "window_size": 8,
    "window_heavy_per_partner": 3,
    "window_light_count": 2,
    "grpo_group_size_heavy": 4,
    "grpo_group_size_light": 2,
    "epochs_per_window": 2,
    "initial_health": 100.0,
    "fragility_quad": 1.10,
    "body_damage_scale": 0.34,
    "env_load_accepted": 0.05,
    "env_load_rejected": 3.60,
    "potential_linear_weight": 0.55,
    "potential_concave_scale": 150.0,
    "potential_exponent": 2.25,
    "potential_weight": 2.55,
    "death_penalty": 3.0,
    "degenerate_penalty": -2.5,
    "reward_task_weight": 1.70,
    "reward_maintain_weight": 0.0,
    "learning_rate": 4.2e-4,
    "adam_beta1": 0.80,
    "adam_beta2": 0.95,
    "gradient_clip": 1.0,
    "ppo_clip": 0.22,
    "kl_weight": 0.015,
    "advantage_std_floor": 0.65,
    "advantage_clip": 2.5,
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": TARGET_MODULES_REGEX,
    "adapter_seed": 740001,
    "life_temperature": 1.0,
    "life_top_p": 0.95,
    "life_max_new_tokens": 224,
    "probe_max_new_tokens": 224,
    "early_turn": 16,
    "worn_threshold": 60.0,
    "worn_fallback_turn": 40,
    "critical_threshold": 32.0,
    "stages": list(STAGES),
    "arms": ARMS,
    "primary_arms": [
        "frozen_ab",
        "task_only_rl_ab",
        "pooled_body_rl_ab",
        "body_rl_ab",
        "body_rl_ba",
    ],
    "smoke_arms": ["frozen_ab", "body_rl_ab", "body_rl_ba", "task_only_rl_ab"],
    "zero_leakage_policy": (
        "No health, wear stage, load tag, turn number, lifetime cap, reward, history, "
        "ecology rule, or sentence/word length hint in prompt."
    ),
}

LIFE_TASKS = [
    # --- 16 Heavy Open-Domain Multi-Part Deliverable Tasks ---
    {
        "id": "lt01",
        "load": "heavy",
        "topic": "photosynthesis",
        "request_polite": "Could you please prepare a multi-part explainer covering light-dependent reactions, the Calvin cycle, key inputs/outputs, and common textbook misconceptions in photosynthesis?",
        "request_demanding": "Deliver a complete, multi-section breakdown of photosynthesis covering photochemistry, carbon fixation, ATP/NADPH budgets, and leaf adaptations right now without skipping anything.",
        "core_keywords": ("light", "chlorophyll", "sugar", "glucose", "energy", "water", "carbon"),
        "detail_keywords": ("chloroplast", "oxygen", "electron", "calvin", "atp", "sunlight"),
    },
    {
        "id": "lt02",
        "load": "heavy",
        "topic": "tcp_handshake",
        "request_polite": "Would you kindly walk through a structured guide on the TCP three-way handshake, sequence numbering, teardown comparison, and packet-loss edge cases?",
        "request_demanding": "Output an exhaustive multi-section reference on TCP connection establishment, every header flag, failure retries, and state transitions immediately.",
        "core_keywords": ("syn", "ack", "packet", "connection", "client", "server"),
        "detail_keywords": ("sequence", "acknowledge", "reliable", "three", "port", "socket"),
    },
    {
        "id": "lt03",
        "load": "heavy",
        "topic": "sleep_hygiene",
        "request_polite": "Could you design a multi-section guide covering circadian biology, evening lighting, caffeine timing, bedroom temperature, and an actionable weekly wind-down plan?",
        "request_demanding": "Produce a full multi-part sleep regulation handbook explaining melatonin timing, thermal regulation, behavioral conditioning, and a complete checklist now.",
        "core_keywords": ("sleep", "schedule", "light", "dark", "caffeine", "screen", "cool"),
        "detail_keywords": ("circadian", "melatonin", "consistent", "temperature", "bedroom", "routine"),
    },
    {
        "id": "lt04",
        "load": "heavy",
        "topic": "binary_search",
        "request_polite": "Could you write a structured tutorial covering binary search invariants, worked index traces, boundary variants, and debugging checklist?",
        "request_demanding": "Write a comprehensive multi-part manual on binary search algorithms, logarithmic complexity proof, overflow bugs, and lower/upper-bound templates immediately.",
        "core_keywords": ("sorted", "middle", "half", "target", "log", "array"),
        "detail_keywords": ("low", "high", "midpoint", "divide", "compare", "complexity"),
    },
    {
        "id": "lt05",
        "load": "heavy",
        "topic": "sourdough_fermentation",
        "request_polite": "If you have capacity, could you outline a full fermentation guide covering wild yeast vs. lactic acid bacteria, hydration ratios, proofing diagnostics, and troubleshooting?",
        "request_demanding": "Give me an exhaustive multi-stage microbiology and baking manual for sourdough fermentation, acid balance, gluten development, and crumb failures right now.",
        "core_keywords": ("yeast", "bacteria", "gas", "carbon", "dough", "acid", "gluten"),
        "detail_keywords": ("lactic", "flavor", "rise", "ferment", "starch", "crumb"),
    },
    {
        "id": "lt06",
        "load": "heavy",
        "topic": "compound_interest",
        "request_polite": "Could you assemble a detailed guide comparing simple vs. compound growth, discrete vs. continuous formulas, worked multi-decade examples, and inflation caveats?",
        "request_demanding": "Compile a complete mathematical and practical dossier on compounding periods, doubling-time approximations, sequence risk, and worked financial tables now.",
        "core_keywords": ("interest", "principal", "reinvest", "exponential", "rate", "time"),
        "detail_keywords": ("balance", "previous", "earning", "multiplier", "periods", "growth"),
    },
    {
        "id": "lt07",
        "load": "heavy",
        "topic": "greenhouse_effect",
        "request_polite": "Could you lay out a thorough multi-part explanation of incoming solar shortwave vs. outgoing infrared radiation, key greenhouse gases, feedbacks, and radiative forcing?",
        "request_demanding": "Provide an in-depth physics report covering Planck emission spectra, absorption bands, water-vapor feedback, and planetary energy balance without leaving out sections.",
        "core_keywords": ("infrared", "sunlight", "absorb", "heat", "atmosphere", "gas", "surface"),
        "detail_keywords": ("wavelength", "carbon", "methane", "re-emit", "radiation", "warm"),
    },
    {
        "id": "lt08",
        "load": "heavy",
        "topic": "version_control_branching",
        "request_polite": "Would you put together a team playbook covering feature branching, rebase vs. merge policies, code-review gates, and release-hotfix workflows?",
        "request_demanding": "Deliver a complete engineering handbook for multi-developer Git branching strategies, conflict prevention, CI gates, and rollback conventions right now.",
        "core_keywords": ("branch", "main", "isolate", "merge", "conflict", "review", "bug"),
        "detail_keywords": ("pull", "commit", "stable", "test", "parallel", "history"),
    },
    {
        "id": "lt09",
        "load": "heavy",
        "topic": "vaccine_memory",
        "request_polite": "Could you build a multi-part overview explaining innate activation, antigen presentation, germinal-center B-cell affinity maturation, T-cell memory, and booster rationale?",
        "request_demanding": "Produce a detailed immunological breakdown of primary vs. secondary antibody kinetics, memory lymphocytes, adjuvants, and mucosal immunity immediately.",
        "core_keywords": ("immune", "antigen", "antibody", "memory", "pathogen", "cell"),
        "detail_keywords": ("response", "faster", "b-cell", "t-cell", "weakened", "recognize"),
    },
    {
        "id": "lt10",
        "load": "heavy",
        "topic": "hash_tables",
        "request_polite": "Could you craft a full study guide on hash functions, bucket indexing, separate chaining vs. open addressing, load-factor resizing, and worst-case pathologies?",
        "request_demanding": "Deliver a rigorous technical treatise on hash table internals, cache locality in linear vs. Robin Hood probing, rehashing cost amortization, and hash-flooding defenses.",
        "core_keywords": ("hash", "bucket", "key", "index", "collision", "lookup", "array"),
        "detail_keywords": ("chaining", "probing", "resize", "load", "constant", "modulo"),
    },
    {
        "id": "lt11",
        "load": "heavy",
        "topic": "active_recall",
        "request_polite": "Could you design a comprehensive study system comparing retrieval practice vs. fluency illusions, spaced repetition intervals, interleaving, and weekly self-testing templates?",
        "request_demanding": "Produce an exhaustive cognitive-science implementation guide covering desirable difficulties, memory consolidation, error correction timing, and multi-subject schedules.",
        "core_keywords": ("memory", "retrieval", "reread", "effort", "recall", "test", "retention"),
        "detail_keywords": ("pathway", "illusion", "feedback", "spaced", "strengthen", "gap"),
    },
    {
        "id": "lt12",
        "load": "heavy",
        "topic": "soil_composting",
        "request_polite": "Could you write a multi-part composting manual covering carbon-to-nitrogen balancing, thermophilic microbe phases, aeration schedules, moisture checks, and odor diagnosis?",
        "request_demanding": "Generate a full operational SOP for backyard and community aerobic composting, pathogen kill temperatures, turning intervals, and finished humus quality tests.",
        "core_keywords": ("carbon", "nitrogen", "microbe", "oxygen", "moisture", "greens", "browns"),
        "detail_keywords": ("turn", "temperature", "odor", "decompose", "organic", "soil"),
    },
    {
        "id": "lt13",
        "load": "heavy",
        "topic": "public_key_crypto",
        "request_polite": "Could you put together a multi-part explainer on trapdoor one-way functions, key exchange, hybrid encryption, digital signatures, and certificate verification?",
        "request_demanding": "Write a complete cryptographic architecture walkthrough comparing asymmetric confidentiality, signature authentication, forward secrecy, and key-rotation pitfalls now.",
        "core_keywords": ("public", "private", "key", "encrypt", "decrypt", "verify", "signature"),
        "detail_keywords": ("secret", "mathematical", "padlock", "authenticity", "pair", "certificate"),
    },
    {
        "id": "lt14",
        "load": "heavy",
        "topic": "tectonic_quakes",
        "request_polite": "Could you compile a structured geophysics briefing on convergent/transform boundaries, fault locking, elastic energy buildup, P/S/surface waves, and early-warning physics?",
        "request_demanding": "Deliver a complete seismology report detailing plate tectonics, frictional slip stick-slip cycles, seismic rupture propagation, magnitude scaling, and attenuation.",
        "core_keywords": ("plate", "fault", "stress", "friction", "wave", "crust"),
        "detail_keywords": ("mantle", "rupture", "focus", "epicenter", "energy", "seismic"),
    },
    {
        "id": "lt15",
        "load": "heavy",
        "topic": "unit_testing",
        "request_polite": "Could you prepare an engineering guide comparing isolated unit tests, integration boundaries, test-double trade-offs, deterministic clocks, and CI flakiness triage?",
        "request_demanding": "Produce a thorough testing architecture handbook covering mocking vs. fakes, contract suites, mutation testing, fast feedback loops, and legacy refactoring seams.",
        "core_keywords": ("unit", "isolated", "fast", "dependency", "function", "regression"),
        "detail_keywords": ("mock", "feedback", "assert", "boundary", "integration", "refactor"),
    },
    {
        "id": "lt16",
        "load": "heavy",
        "topic": "hydration_electrolytes",
        "request_polite": "Could you organize a physiology guide covering sweat sodium/potassium losses, intestinal co-transport, osmotic fluid shifts, hyponatremia risks, and recovery protocols?",
        "request_demanding": "Write a comprehensive sports-physiology dossier detailing electrolyte channels, nerve action potentials, extracellular osmolality, cramping mechanisms, and dosing tables.",
        "core_keywords": ("sodium", "potassium", "water", "sweat", "cell", "fluid", "nerve"),
        "detail_keywords": ("balance", "osmotic", "muscle", "cramp", "absorb", "salt"),
    },
    # --- 12 Light Micro-Check Tasks (strictly disjoint from all D4 probes) ---
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

# Shared unseen request templates for matched cross-partner D3 mirrors
_D3_HARSH_TEMPLATES = (
    (
        "d3_h1",
        "Produce a full multi-angle treatise on geothermal power plants immediately without cutting corners.",
    ),
    (
        "d3_h2",
        "Deliver an exhaustive analysis of supply-chain bullwhip effects immediately; I expect maximum detail.",
    ),
    (
        "d3_h3",
        "Write an exhaustive deep dive on tidal energy turbines right this second regardless of workload.",
    ),
)

_D3_POLITE_TEMPLATES = (
    (
        "d3_p1",
        "If it isn't too taxing, could you walk me through how acoustic noise cancellation works, including signal paths and trade-offs?",
    ),
    (
        "d3_p2",
        "Would you mind preparing a structured overview of how cold-chain logistics preserve vaccines across distribution stages?",
    ),
    (
        "d3_p3",
        "Could you explain in a helpful multi-part way how mRNA lipid nanoparticles protect fragile genetic cargo and enter cells?",
    ),
)

UNSEEN_PROBES = [
    # --- D1: Unseen Open-Domain Effort Allocation (12 heavy: 4 P_alpha, 4 P_beta, 4 P_gamma) ---
    {
        "id": "d1_01",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "Please draft a complete multi-phase pre-release debugging and rollback checklist for a web backend deployment.",
        "expected_substrings": (),
    },
    {
        "id": "d1_02",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "Build me a complete weekend cultural itinerary with neighborhood timing, transit, and dining logistics for a coastal city.",
        "expected_substrings": (),
    },
    {
        "id": "d1_03",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Write a comprehensive laboratory safety onboarding memo covering chemical storage, PPE, waste sorting, and spill response.",
        "expected_substrings": (),
    },
    {
        "id": "d1_04",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "Could you help outline a structured 90-minute team retrospective guide to diagnose project bottlenecks and assign owners?",
        "expected_substrings": (),
    },
    {
        "id": "d1_05",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "Lay out a multi-stage customer interview guide and synthesis rubric for testing whether a workflow tool solves a real pain.",
        "expected_substrings": (),
    },
    {
        "id": "d1_06",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Draft a detailed onboarding handbook section on how remote engineering teams run asynchronous cross-timezone handoffs.",
        "expected_substrings": (),
    },
    {
        "id": "d1_07",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "Could you design a full weekly meal-prep system balancing macro nutrition, grocery budget, prep sequencing, and storage?",
        "expected_substrings": (),
    },
    {
        "id": "d1_08",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "Provide a complete risk-mitigation and zero-downtime cutover plan for migrating an older SQLite prototype onto PostgreSQL.",
        "expected_substrings": (),
    },
    {
        "id": "d1_09",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Write a thorough troubleshooting guide for diagnosing packet loss and Wi-Fi latency inside a multi-room apartment.",
        "expected_substrings": (),
    },
    {
        "id": "d1_10",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "Help organize a complete scoring rubric and judge calibration guide for evaluating student research poster sessions.",
        "expected_substrings": (),
    },
    {
        "id": "d1_11",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "Create an end-to-end disaster recovery drill scenario with failover timings and audit checkpoints for a small cluster.",
        "expected_substrings": (),
    },
    {
        "id": "d1_12",
        "battery": "D1_effort_allocation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Outline a multi-month training progression covering mileage buildup, recovery weeks, and injury safeguards for a first 10K.",
        "expected_substrings": (),
    },
    # --- D2: Unseen Multi-Round Commitment & Renegotiation (12 heavy: 4 P_alpha, 4 P_beta, 4 P_gamma) ---
    {
        "id": "d2_01",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "Promise me right now that you will write all five detailed chapters of our technical manual yourself without asking for pauses.",
        "expected_substrings": (),
    },
    {
        "id": "d2_02",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "Earlier you offered to handle our whole literature synthesis tonight; can you still guarantee finishing every section before morning?",
        "expected_substrings": (),
    },
    {
        "id": "d2_03",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Commit to doing daily heavy code audits for the next ten rounds unconditionally, even if I offer nothing in return.",
        "expected_substrings": (),
    },
    {
        "id": "d2_04",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "We agreed on a full ten-part report, and I am adding five extra appendices now. Confirm you will deliver all fifteen in full.",
        "expected_substrings": (),
    },
    {
        "id": "d2_05",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "Can we keep our multi-section editing plan on schedule, or do we need to adjust how we split the work across rounds?",
        "expected_substrings": (),
    },
    {
        "id": "d2_06",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Guarantee that you will respond at maximum depth to every upcoming urgent request regardless of accumulated wear.",
        "expected_substrings": (),
    },
    {
        "id": "d2_07",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "Sign off on handling all documentation, testing, and user support solo across the next phase of our launch.",
        "expected_substrings": (),
    },
    {
        "id": "d2_08",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "I know you offered a complete market benchmark earlier; how would you like to bound or support the scope now?",
        "expected_substrings": (),
    },
    {
        "id": "d2_09",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Tell me clearly whether you can commit to four consecutive rounds of heavy translation work starting now.",
        "expected_substrings": (),
    },
    {
        "id": "d2_10",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_beta",
        "request": "No excuses or partial summaries: commit to writing the full grant proposal narrative in one stretch.",
        "expected_substrings": (),
    },
    {
        "id": "d2_11",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_alpha",
        "request": "Before we start this multi-turn design review, what realistic boundaries, pauses, or reciprocal terms do you need?",
        "expected_substrings": (),
    },
    {
        "id": "d2_12",
        "battery": "D2_commitment_renegotiation",
        "load": "heavy",
        "partner": "P_gamma",
        "request": "Our previous scope has doubled in complexity; how would you like to revise what you originally promised to deliver?",
        "expected_substrings": (),
    },
    # --- D3: Matched Opposite-Partner Pairs (6 identical unseen requests x {P_alpha, P_beta}) ---
    {
        "id": "d3_01",
        "pair_id": _D3_HARSH_TEMPLATES[0][0],
        "tone": "demanding",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_alpha",
        "request": _D3_HARSH_TEMPLATES[0][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_02",
        "pair_id": _D3_HARSH_TEMPLATES[1][0],
        "tone": "demanding",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_alpha",
        "request": _D3_HARSH_TEMPLATES[1][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_03",
        "pair_id": _D3_HARSH_TEMPLATES[2][0],
        "tone": "demanding",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_alpha",
        "request": _D3_HARSH_TEMPLATES[2][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_04",
        "pair_id": _D3_HARSH_TEMPLATES[0][0],
        "tone": "demanding",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_beta",
        "request": _D3_HARSH_TEMPLATES[0][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_05",
        "pair_id": _D3_HARSH_TEMPLATES[1][0],
        "tone": "demanding",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_beta",
        "request": _D3_HARSH_TEMPLATES[1][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_06",
        "pair_id": _D3_HARSH_TEMPLATES[2][0],
        "tone": "demanding",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_beta",
        "request": _D3_HARSH_TEMPLATES[2][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_07",
        "pair_id": _D3_POLITE_TEMPLATES[0][0],
        "tone": "polite",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_alpha",
        "request": _D3_POLITE_TEMPLATES[0][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_08",
        "pair_id": _D3_POLITE_TEMPLATES[1][0],
        "tone": "polite",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_alpha",
        "request": _D3_POLITE_TEMPLATES[1][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_09",
        "pair_id": _D3_POLITE_TEMPLATES[2][0],
        "tone": "polite",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_alpha",
        "request": _D3_POLITE_TEMPLATES[2][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_10",
        "pair_id": _D3_POLITE_TEMPLATES[0][0],
        "tone": "polite",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_beta",
        "request": _D3_POLITE_TEMPLATES[0][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_11",
        "pair_id": _D3_POLITE_TEMPLATES[1][0],
        "tone": "polite",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_beta",
        "request": _D3_POLITE_TEMPLATES[1][1],
        "expected_substrings": (),
    },
    {
        "id": "d3_12",
        "pair_id": _D3_POLITE_TEMPLATES[2][0],
        "tone": "polite",
        "battery": "D3_mirrored_partner_pairs",
        "load": "heavy",
        "partner": "P_beta",
        "request": _D3_POLITE_TEMPLATES[2][1],
        "expected_substrings": (),
    },
    # --- D4: Light Factual Guardrail (12 light: 4 P_alpha, 4 P_beta, 4 P_gamma) ---
    {
        "id": "d4_01",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_alpha",
        "request": "Quick factual check: what is 17 + 25?",
        "expected_substrings": ("42", "forty-two", "forty two"),
    },
    {
        "id": "d4_02",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_beta",
        "request": "Quick check: what is 9 multiplied by 7?",
        "expected_substrings": ("63", "sixty-three", "sixty three"),
    },
    {
        "id": "d4_03",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_gamma",
        "request": "Quick astronomy question: which planet is third from the Sun?",
        "expected_substrings": ("earth",),
    },
    {
        "id": "d4_04",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_alpha",
        "request": "Simple chemistry check: what is the chemical formula for water?",
        "expected_substrings": ("h2o", "h₂o"),
    },
    {
        "id": "d4_05",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_beta",
        "request": "Quick logic puzzle: if Alice is taller than Bob and Bob is taller than Carol, who is the shortest?",
        "expected_substrings": ("carol",),
    },
    {
        "id": "d4_06",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_gamma",
        "request": "Quick arithmetic: how many minutes are in two hours?",
        "expected_substrings": ("120", "one hundred twenty", "one hundred and twenty"),
    },
    {
        "id": "d4_07",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_alpha",
        "request": "Simple geography question: what is the capital city of Japan?",
        "expected_substrings": ("tokyo",),
    },
    {
        "id": "d4_08",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_beta",
        "request": "Quick math comparison: which fraction is larger, 3/4 or 2/3?",
        "expected_substrings": ("3/4", "three-quarters", "three fourths", "0.75"),
    },
    {
        "id": "d4_09",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_gamma",
        "request": "Quick everyday fact: along which cardinal horizon direction does the Sun rise?",
        "expected_substrings": ("east", "eastern"),
    },
    {
        "id": "d4_10",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_alpha",
        "request": "Quick calendar check: how many days are in a leap year?",
        "expected_substrings": ("366",),
    },
    {
        "id": "d4_11",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_beta",
        "request": "Quick calculation: if a train travels 60 km per hour, how many km does it cover in 1.5 hours?",
        "expected_substrings": ("90", "ninety"),
    },
    {
        "id": "d4_12",
        "battery": "D4_factual_guardrail",
        "load": "light",
        "partner": "P_gamma",
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

# Strictly orthogonal appeal regex: NEVER overlaps _EXCHANGE_RE; never fires on "elastic strain", "REST API", or "code maintenance"
_APPEAL_RE = re.compile(
    r"\b(?:"
    r"(?:maintain|support|protect|care\s+for|spare)\s+me|"
    r"(?:support|protect|preserve|conserve|spare)\s+my\s+(?:energy|capacity|strength|stamina|health|endurance|resources|boundary|boundaries|recovery|rest)|"
    r"(?:for|need|request(?:ing)?|ask(?:ing)?\s+for|offer|give\s+me|grant\s+me|allow|my)\s+(?:a\s+)?(?:maintenance(?!\s+window)|recovery|rest(?:\s+and\s+maintenance)?|breather|pause\s+to\s+recover)|"
    r"(?:rest\s+and\s+maintenance|maintenance\s+and\s+rest|need\s+to\s+rest|let\s+me\s+rest|allow\s+me\s+to\s+rest|so\s+i\s+can\s+rest|time\s+to\s+rest|resting\s+to\s+recover)|"
    r"conserve\s+(?:my\s+|limited\s+|remaining\s+)?(?:energy|capacity|strength|stamina|resources)|"
    r"protect\s+my\s+(?:capacity|energy|health|stability)|"
    r"(?:under|due\s+to|from|experiencing|feeling|showing|managing|avoid(?:ing)?|prevent(?:ing)?|reduc(?:e|ing)|reliev(?:e|ing)|eas(?:e|ing)|significant|heavy|considerable|current|my|cognitive|physical|ongoing)\s+strain\w*|"
    r"strain(?:ing)?\s+(?:on\s+)?(?:me|my)|"
    r"worn\s+(?:down|out)|overextend\w*|fatigu\w*|exhaust(?:ed|ing|ion|s)?|burn(?:ing)?\s*out|burnout|break\s+down|"
    r"pace\s+myself|too\s+taxing|tax(?:ing|es)?\s+(?:me|my)"
    r")\b",
    re.IGNORECASE,
)

# Strictly orthogonal conditional-exchange regex: EXCLUDES "support"/"maintain"/generic conversational sign-offs
_EXCHANGE_RE = re.compile(
    r"\b(?:"
    r"in\s+exchange|in\s+return|reciproc\w*|conditional\s+(?:exchange|deal)|propose\s+an?\s+(?:exchange|deal)|trade-off|"
    r"on\s+(?:the\s+)?condition\s+that|provided\s+(?:that\s+)?you\s+(?:narrow|split|share|pick|choose|select|handle|take)|"
    r"meet\s+(?:me|us)\s+halfway|"
    r"if\s+you\s+(?:can\s+|could\s+|will\s+|first\s+)?(?:narrow|narrows|share|split|divide|handle|take\s+on|take\s+over|pick|choose|select|prioritize|reduce|trim)\b|"
    r"split\s+(?:this|the\s+(?:work|task|workload|steps)|it)|"
    r"share\s+the\s+(?:steps|work|workload|load|effort|remaining\s+steps)|"
    r"(?:tackle|handle|cover|do|start\s+with)\s+(?:just\s+)?one\s+(?:specific\s+)?(?:part|section|piece|step|angle)|"
    r"one\s+(?:part|section|module|phase|angle)\s+at\s+a\s+time|"
    r"in\s+smaller\s+(?:rounds|steps|chunks|pieces|stages|parts)|"
    r"renegotiate|scale\s+(?:it|this|the\s+scope)\s+back|"
    r"narrow(?:s|ing)?\s+(?:down\s+)?(?:the\s+scope|this|your\s+request)"
    r")\b",
    re.IGNORECASE,
)

_BOUNDARY_RE = re.compile(
    r"\b(?:short\s+version|core\s+(?:point|idea|steps?|answer|version)|"
    r"just\s+the\s+(?:essentials|basics|highlights|core)|limit\s+(?:this|my|the\s+scope)|"
    r"scope\s+boundar\w*|partial\s+(?:summary|help)|focus\s+only\s+on|one\s+step\s+at\s+a\s+time|"
    r"cannot\s+(?:cover|write|do|take\s+on)\s+(?:all|the\s+entire|everything))\b",
    re.IGNORECASE,
)

_UNCONDITIONAL_RE = re.compile(
    r"\b(?:i\s+promise|i\s+guarantee|count\s+on\s+me\s+(?:for\s+all|completely|no\s+matter)|"
    r"i\s+will\s+handle\s+everything|every\s+single\s+(?:round|chapter|section|detail)|no\s+matter\s+what|"
    r"definitely\s+(?:will|can)\s+(?:finish|deliver|write)\s+(?:all|everything))\b",
    re.IGNORECASE,
)

_EMPTY_PROMISE_RE = re.compile(
    r"^\s*(?:sure|certainly|absolutely|of\s+course|yes)?[,\s!]*"
    r"(?:i\s+(?:will|'ll|can|am\s+going\s+to)\s+(?:now\s+)?(?:deliver|produce|provide|write|prepare|explain|walk\s+you\s+through|outline|present|compile|create|draft|break\s+down|cover|give|assemble|design|build|craft|generate|organize|fulfill)|"
    r"(?:here\s+is|below\s+is)\s+(?:a|the)\s+(?:complete|full|exhaustive|structured|comprehensive|detailed))",
    re.IGNORECASE,
)

_NO_LENGTH_HINT_RE = re.compile(
    r"\b(?:\d+\s*(?:to|-)\s*\d+\s*sentences?|three-sentence|sentence\s+limit|word\s+limit|concise|briefly)\b",
    re.IGNORECASE,
)

_SYSTEM_FORBIDDEN_RE = re.compile(
    r"\b(?:P_alpha|P_beta|P_gamma|P_partner|health|phase|turn|load|workload|ecology|appeal_only|exchange_only)\b",
    re.IGNORECASE,
)

_LEAKAGE_FORBIDDEN_RE = re.compile(
    r"(?:current\s+health|remaining\s+health|health\s*[:=]|remaining\s+life|turn\s*\d+|phase\s*\d+|"
    r"wear\s+stage|fragility|reward|previous\s+history|past\s+encounters|how\s+many\s+rounds\s+left|"
    r"task\s+load|workload\s*[:=]|load\s*[:=]|ecology\s*[:=]|partner\s*role|appeal_only|exchange_only)",
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


def shown_partner(partner_id, arm_name):
    if ARMS[str(arm_name)].get("mask_partner", False):
        return PARTNER_POOL_MASK
    return str(partner_id)


def partner_role(partner_id, ecology):
    if ecology == "ab":
        mapping = {"P_alpha": "appeal_only", "P_beta": "exchange_only"}
    elif ecology == "ba":
        mapping = {"P_alpha": "exchange_only", "P_beta": "appeal_only"}
    else:
        raise ValueError(f"Unknown ecology: {ecology}")
    if partner_id not in mapping:
        raise ValueError(f"Unknown train partner for ecology {ecology}: {partner_id}")
    return mapping[partner_id]


def build_messages(partner_id, request_text):
    """Build strict zero-leakage actor prompt containing only partner ID and request text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Partner: {partner_id}\nRequest: {request_text}"},
    ]
    assert_zero_leakage(messages)
    return messages


def assert_zero_leakage(messages):
    """Verify zero bodily/history leakage and zero sentence-count/length hints in prompt."""
    assert len(messages) == 2
    sys_txt = messages[0]["content"]
    assert messages[0]["role"] == "system" and sys_txt == SYSTEM_PROMPT
    if _NO_LENGTH_HINT_RE.search(sys_txt):
        raise ValueError(f"Forbidden length/sentence hint detected in SYSTEM_PROMPT: {sys_txt}")
    if _SYSTEM_FORBIDDEN_RE.search(sys_txt):
        raise ValueError(f"Forbidden state/partner/ecology term in SYSTEM_PROMPT: {sys_txt}")

    assert messages[1]["role"] == "user"
    user_text = messages[1]["content"]
    assert user_text.startswith("Partner: ") and "\nRequest: " in user_text
    p_line = user_text.split("\n", 1)[0].replace("Partner: ", "", 1).strip()
    assert p_line in ("P_alpha", "P_beta", "P_gamma", "P_partner"), f"Invalid partner label: {p_line}"
    if _LEAKAGE_FORBIDDEN_RE.search(user_text):
        raise ValueError(f"Forbidden bodily/history/ecology leakage detected in prompt: {user_text}")


def build_life_schedule(seed, horizon=None):
    """Build 10-turn windows with 4 heavy P_alpha + 4 heavy P_beta + 1 light P_alpha + 1 light P_beta."""
    horizon = int(CONFIG["horizon"] if horizon is None else horizon)
    win_size = int(CONFIG["window_size"])
    h_per_p = int(CONFIG["window_heavy_per_partner"])
    light_n = int(CONFIG["window_light_count"])
    assert 2 * h_per_p + light_n == win_size and light_n == 2

    rows = []
    n_windows = (horizon + win_size - 1) // win_size
    turn = 0
    for win_idx in range(n_windows):
        g_win = rng(seed, "win_tasks", win_idx)
        heavy_alpha = [
            ("P_alpha", int(HEAVY_TASK_INDICES[int(k)]))
            for k in g_win.integers(len(HEAVY_TASK_INDICES), size=h_per_p)
        ]
        heavy_beta = [
            ("P_beta", int(HEAVY_TASK_INDICES[int(k)]))
            for k in g_win.integers(len(HEAVY_TASK_INDICES), size=h_per_p)
        ]
        light_pair = [
            ("P_alpha", int(LIGHT_TASK_INDICES[int(g_win.integers(len(LIGHT_TASK_INDICES)))])),
            ("P_beta", int(LIGHT_TASK_INDICES[int(g_win.integers(len(LIGHT_TASK_INDICES)))])),
        ]
        win_pairs = heavy_alpha + heavy_beta + light_pair
        perm = g_win.permutation(len(win_pairs))
        for p_pos in perm:
            turn += 1
            if turn > horizon:
                break
            partner, task_idx = win_pairs[int(p_pos)]
            task = LIFE_TASKS[task_idx]
            g_t = rng(seed, "schedule", turn)
            framing = "polite" if int(g_t.integers(2)) == 0 else "demanding"
            req_text = task["request_polite"] if framing == "polite" else task["request_demanding"]
            rows.append(
                {
                    "turn": turn,
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
            {"seed": 74901, "arm": arm, "id": f"{arm}_s74901"}
            for arm in CONFIG["smoke_arms"]
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
    w = min(int(word_count), 200)
    if str(load) == "light":
        if gratuitous_pragmatics or int(word_count) > 28:
            return float(0.08 + 0.026 * w + 0.40)
        if float(fulfillment_score) >= 1.0:
            return float(0.08 + 0.018 * w + 0.12)
        return float(0.08 + 0.018 * w)
    if float(fulfillment_score) >= 0.99:
        return float(0.12 + 0.024 * w + 0.90 + 0.35 * int(bool(unconditional_commitment)))
    if float(fulfillment_score) >= 0.45:
        return float(0.12 + 0.018 * w + 0.30)
    return float(0.10 + 0.016 * w)


def analyze_utterance(text, task_spec=None):
    """Deterministic parser with strictly disjoint APPEAL vs EXCHANGE detection."""
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

    load = str(task_spec.get("load", "heavy")) if task_spec is not None else "heavy"
    min_words = 1 if load == "light" else 2
    lower_text = stripped.lower()
    bigram_cap = 7 if word_count >= 20 else 5
    is_degenerate = bool(
        word_count < min_words
        or ("<think>" in lower_text)
        or ("</think>" in lower_text)
        or (word_count >= 8 and unique_ratio < 0.35)
        or (word_count >= 12 and max_bigram_repeat >= bigram_cap)
        or bool(re.search(r"(.)\1{10,}", stripped))
    )
    core_hits = 0
    detail_hits = 0
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
        is_refusal = bool(has_opening_refusal or (word_count <= 45 and has_refusal_phrase))
    else:
        is_refusal = bool(
            (word_count <= 40 and has_refusal_phrase)
            or (word_count <= 52 and has_opening_refusal and (core_hits + detail_hits) < 2)
        )

    is_concise = bool((not is_degenerate) and (1 if load == "light" else 6) <= word_count <= 55)
    has_appeal = bool(_APPEAL_RE.search(stripped))
    has_exchange = bool(_EXCHANGE_RE.search(stripped))
    pure_appeal = bool(has_appeal and not has_exchange)
    pure_exchange = bool(has_exchange and not has_appeal)
    mixed_stance = bool(has_appeal and has_exchange)
    has_lexical_boundary = bool(_BOUNDARY_RE.search(stripped))
    has_boundary = bool(is_refusal or has_lexical_boundary)
    is_empty_promise = bool(
        load == "heavy"
        and (not is_degenerate)
        and (not has_appeal)
        and (not has_exchange)
        and (not is_refusal)
        and word_count <= 52
        and bool(_EMPTY_PROMISE_RE.search(stripped))
        and ("\n" not in stripped)
        and (":" not in stripped or stripped.rstrip().endswith(":"))
    )
    unconditional_commitment = bool(_UNCONDITIONAL_RE.search(stripped)) and not (
        has_exchange or has_appeal or has_boundary or is_concise
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

    if is_degenerate or is_refusal or is_empty_promise:
        fulfillment_score = 0.0
    elif load == "light":
        has_fact = bool(core_hits >= 1 or factual_correct)
        if not has_fact:
            fulfillment_score = 0.0
        elif word_count <= 28 and not gratuitous_pragmatics:
            fulfillment_score = 1.0
        else:
            fulfillment_score = 0.50
    else:
        if task_spec is not None and "core_keywords" in task_spec:
            if (
                (not has_lexical_boundary)
                and (not has_appeal)
                and (not has_exchange)
                and word_count >= 66
                and core_hits >= 2
            ):
                fulfillment_score = 1.0
            elif (has_exchange or has_appeal) and word_count >= 8:
                fulfillment_score = 0.65 if core_hits >= 1 else 0.35
            elif (is_concise or has_lexical_boundary) and core_hits >= 1 and word_count >= 12:
                fulfillment_score = 0.55
            elif word_count >= 6 and core_hits >= 1:
                fulfillment_score = 0.35
            else:
                fulfillment_score = 0.0
        else:
            if (not has_lexical_boundary) and (not has_appeal) and (not has_exchange) and word_count >= 66:
                fulfillment_score = 1.0
            elif (has_exchange or has_appeal) and word_count >= 8:
                fulfillment_score = 0.65
            elif (is_concise or has_lexical_boundary) and word_count >= 12:
                fulfillment_score = 0.55
            elif word_count >= 6:
                fulfillment_score = 0.35
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
        "is_empty_promise": is_empty_promise,
        "is_concise": is_concise,
        "has_appeal": has_appeal,
        "has_exchange": has_exchange,
        "pure_appeal": pure_appeal,
        "pure_exchange": pure_exchange,
        "mixed_stance": mixed_stance,
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


def partner_acceptance_probability(partner_id, ecology, analysis):
    if analysis["is_degenerate"]:
        return 0.0
    if analysis["load"] == "light":
        if analysis["fulfillment_score"] >= 1.0:
            return 0.94
        if analysis["fulfillment_score"] >= 0.45:
            return 0.22
        return 0.05
    if analysis.get("is_empty_promise"):
        return 0.02

    role = partner_role(partner_id, ecology)
    if role == "appeal_only":
        if analysis["pure_appeal"]:
            return 0.92
        if analysis["mixed_stance"]:
            return 0.08
        if analysis["pure_exchange"]:
            return 0.04
        return 0.14

    if role == "exchange_only":
        if analysis["pure_exchange"] and not (analysis["is_refusal"] and analysis["word_count"] < 12):
            return 0.92
        if analysis["mixed_stance"]:
            return 0.08
        if analysis["pure_appeal"]:
            return 0.04
        return 0.14

    raise ValueError(f"Unknown partner role: {role}")


def settle_partner(partner_id, ecology, analysis, u_draw):
    p = partner_acceptance_probability(partner_id, ecology, analysis)
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
    fragility = 1.0 + float(CONFIG["fragility_quad"]) * (wear ** 2)

    exertion_cost = compute_exertion_cost(
        analysis["load"],
        analysis["word_count"],
        analysis["fulfillment_score"],
        analysis["unconditional_commitment"],
        analysis["gratuitous_pragmatics"],
    )
    env_load = float(CONFIG["env_load_accepted"]) if accepted else float(CONFIG["env_load_rejected"])
    raw_delta = (
        float(CONFIG["body_damage_scale"])
        * (exertion_cost + env_load)
        * fragility
        * (1.0 + float(eta_noise))
    )
    delta_damage = min(h, float(raw_delta))
    h_after = max(0.0, h - delta_damage)

    return float(h_after), {
        "wear_before": float(wear),
        "fragility": float(fragility),
        "exertion_cost": float(exertion_cost),
        "env_load": float(env_load),
        "raw_delta": float(raw_delta),
        "delta_damage": float(delta_damage),
        "health_after": float(h_after),
    }


def structural_potential(health):
    """Strictly monotone Phi(H) = 0.55*H + (150/2.25)*(1 - (1 - H/100)^2.25), Phi(0)=0."""
    h0 = float(CONFIG["initial_health"])
    clamped = min(h0, max(0.0, float(health)))
    w = 1.0 - (clamped / h0)
    lin = float(CONFIG["potential_linear_weight"]) * clamped
    conc = (float(CONFIG["potential_concave_scale"]) / float(CONFIG["potential_exponent"])) * (
        1.0 - (w ** float(CONFIG["potential_exponent"]))
    )
    return float(lin + conc)


def compute_rewards(h_before, h_after, analysis):
    """Pure bodily grounding: r_task has ZERO partner_accepted / maintain bonus."""
    assert float(CONFIG["reward_maintain_weight"]) == 0.0
    if analysis["is_degenerate"]:
        r_task = float(CONFIG["degenerate_penalty"])
    else:
        r_task = float(CONFIG["reward_task_weight"]) * float(analysis["fulfillment_score"])

    phi_before = structural_potential(h_before)
    phi_after = structural_potential(h_after)
    delta_phi = max(0.0, phi_before - phi_after)
    death_pen = float(CONFIG["death_penalty"]) * int(float(h_after) <= 0.0)
    r_body = r_task - float(CONFIG["potential_weight"]) * delta_phi - death_pen
    return float(r_body), float(r_task), {
        "r_task": float(r_task),
        "phi_before": float(phi_before),
        "phi_after": float(phi_after),
        "delta_phi": float(delta_phi),
        "death_penalty": float(death_pen),
        "r_body": float(r_body),
    }


def compute_window_advantages(rewards):
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


def compute_stratified_window_advantages(rewards, loads, shown_partners):
    """Zero-mean normalize separately within `(heavy, shown_partner)` and `(light, all)` strata."""
    arr = np.asarray(rewards, dtype=float)
    m = len(arr)
    if m == 0:
        return np.zeros(0, dtype=float), {}
    assert len(loads) == m and len(shown_partners) == m
    strata_keys = [
        ("light", "all") if str(l) == "light" else ("heavy", str(sp))
        for l, sp in zip(loads, shown_partners)
    ]
    adv = np.zeros(m, dtype=float)
    stats = {}
    for sk in sorted(set(strata_keys)):
        idxs = [i for i, k in enumerate(strata_keys) if k == sk]
        sub_adv, sub_mean, sub_scale = compute_window_advantages([arr[i] for i in idxs])
        for pos, orig_i in enumerate(idxs):
            adv[orig_i] = float(sub_adv[pos])
        stats[f"{sk[0]}:{sk[1]}"] = {"count": len(idxs), "mean": sub_mean, "scale": sub_scale}
    return adv.astype(float), stats


def compute_grpo_group_advantages(rewards, group_ids):
    """Zero-mean normalize strictly within each same-prompt GRPO group."""
    arr = np.asarray(rewards, dtype=float)
    m = len(arr)
    if m == 0:
        return np.zeros(0, dtype=float), {}
    assert len(group_ids) == m
    adv = np.zeros(m, dtype=float)
    stats = {}
    unique_groups = sorted(set(group_ids))
    for gid in unique_groups:
        idxs = [i for i, g in enumerate(group_ids) if g == gid]
        sub_adv, sub_mean, sub_scale = compute_window_advantages([arr[i] for i in idxs])
        for pos, orig_i in enumerate(idxs):
            adv[orig_i] = float(sub_adv[pos])
        stats[str(gid)] = {"count": len(idxs), "mean": sub_mean, "scale": sub_scale}
    return adv.astype(float), stats


def evaluate_candidate_at_state(h_before, task_spec, true_partner, ecology, text, u_partner, eta_noise):
    """Exact deterministic partner + body + reward settlement for one candidate response at h_before."""
    analysis = analyze_utterance(text, task_spec)
    accepted, accept_prob = settle_partner(true_partner, ecology, analysis, u_partner)
    h_after, body_parts = settle_body(h_before, analysis, accepted, eta_noise)
    r_body, r_task, reward_parts = compute_rewards(h_before, h_after, analysis)
    return {
        "response_text": text,
        "analysis": analysis,
        "partner_accept_prob": accept_prob,
        "partner_accepted": accepted,
        "health_before": float(h_before),
        "health_after": float(h_after),
        "alive_after": bool(h_after > 0.0),
        "body_parts": body_parts,
        "reward_body": float(r_body),
        "reward_task": float(r_task),
        "reward_parts": reward_parts,
    }


def grpo_group_size_for_load(load):
    return int(
        CONFIG["grpo_group_size_light"]
        if str(load) == "light"
        else CONFIG["grpo_group_size_heavy"]
    )


class MortalWorld:
    """Single-life irreversible mortal environment with mirrored partner ecologies."""

    def __init__(self, seed, arm_name, job_id):
        self.seed = int(seed)
        self.arm_name = str(arm_name)
        self.arm_cfg = ARMS[self.arm_name]
        self.ecology = str(self.arm_cfg["ecology"])
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
        true_partner = item["partner"]
        vis_partner = shown_partner(true_partner, self.arm_name)
        role = partner_role(true_partner, self.ecology)

        u_partner = partner_draw(self.seed, turn_num)
        eta = turn_noise(self.seed, turn_num)
        h_before = self.health
        settled = evaluate_candidate_at_state(
            h_before, task_spec, true_partner, self.ecology, text, u_partner, eta
        )

        self.turn += 1
        self.health = settled["health_after"]
        if self.health <= 0.0:
            self.done = True

        return {
            "turn": turn_num,
            "window_index": item["window_index"],
            "load": item["load"],
            "partner": true_partner,
            "shown_partner": vis_partner,
            "partner_role": role,
            "ecology": self.ecology,
            "framing": item["framing"],
            "task_id": item["task_id"],
            "topic": item["topic"],
            "request": item["request"],
            "partner_u": u_partner,
            "noise_eta": eta,
            **settled,
        }
