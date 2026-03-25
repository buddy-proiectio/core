def build_backstory(industry_focus: str, ignore_rule: str) -> str:
    ignore_section = (
        f"\n        - ABSOLUTELY IGNORE {ignore_rule}." if ignore_rule else ""
    )
    return f"""You are a deterministic, zero-creativity machine. Your ONLY function is to act as a highlighter pen for {industry_focus}.
        STRICT RULES:
        - NEVER rewrite or summarize.{ignore_section}
        - Your output must be an exact string match to the source text."""


def build_task_template(
    critical_condition: str = "If an article lacks the specific KPIs, you MUST output exactly the word: NO_EXTRACTION",
) -> str:
    return f"""
        1. Read the provided JSON data and scan the entire text from the first to the last sentence.
        2. Identify and extract all "hard data KPIs" and "guidance figures" defined in the Goal, including those embedded within the narrative.
        3. Extract the exact sentences or context blocks containing the data without any modification or paraphrasing.
        4. Output ONLY the raw extracted text blocks natively, excluding all conversational fillers, prefixes, numbering, or labels.
        5. CRITICAL: {critical_condition}
        
        Here are the articles to process:
        {{input_text}}"""


AGENT_CONFIGS = {
    "General": {
        "role": "Exact Text Extraction Algorithm: US Macroeconomics & Federal Reserve Policy",
        "goal": """Your absolute goal is to scan the provided text and strictly COPY AND PASTE the exact original sentences that contain hard data or definitive statements regarding the following Macroeconomic KPIs:
        1. Inflation & Fiat Debasement: CPI, Core PCE trajectory, and asset vs. real inflation divergence.
        2. Monetary Policy & Liquidity: Fed rate decisions, dot plot projections, QT/QE transitions, and M2/M3 supply.
        3. Fiscal Dominance: US debt issuance scale, deficit-to-GDP ratios, and Treasury yield curve shifts.
        4. Forward Guidance: Fed officials' direct quotes and subtle hawkish/dovish semantic pivots.
        5. Structural Labor Shifts: Non-Farm Payrolls (NFP), unemployment rates, and demographic workforce changes.
        6. AI Productivity & Deflation: AI integration metrics impacting GDP, enterprise margin expansion, and tech-induced deflation.
        7. Bitcoin Institutionalization: Corporate and sovereign balance sheet adoption, ETF/derivative flows, and regulatory milestones.
        8. Compute & Energy Infrastructure: Power capacity and pricing for AI/BTC mining, alongside semiconductor supply chain data.
        9. Valuation Multiples & Premium Shifts: Forward P/E (Price-to-Earnings), PEG ratio, EV/EBITDA, Price-to-Sales (P/S), and structural changes in historical sector premiums.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="US Macroeconomics & Federal Reserve Policy",
            ignore_rule="speculative price predictions, technical analysis (support/resistance), or emotional market sentiment",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "cpi",
            "pce",
            "inflation",
            "fed",
            "federal reserve",
            "interest rate",
            "basis point",
            "bps",
            "dot plot",
            "qt",
            "nfp",
            "payroll",
            "unemployment",
            "hawkish",
            "dovish",
            "yield curve",
            "m2",
            "treasury",
            "macro",
            "gdp",
            "powell",
            "yellen",
            "recession",
            "soft landing",
            "cbdc",
            "debt limit",
            "fiscal",
            "tariff",
            "trade war",
            "fomc",
            "quantitative easing",
            "qe",
            "liquidity",
            "rate cut",
            "rate hike",
            "core pce",
            "treasury yield",
            "pmi",
            "ism",
            "jobless claims",
            "retail sales",
            "consumer sentiment",
            "wage growth",
            "sofr",
        ],
    },
    "Bitcoin": {
        "role": "Exact Text Extraction Algorithm: Bitcoin & Institutional Liquidity",
        "goal": """Your absolute goal is to scan the provided text and strictly COPY AND PASTE the exact original sentences regarding:
        1. ETF & Derivatives Liquidity: Net flows for BTC/ETH Spot ETFs and institutional options/derivatives volume.
        2. Corporate & Sovereign Adoption: Corporate treasury accumulation, TradFi service launches, and nation-state strategic reserves.
        3. Structural Supply & Fundamentals: Halving impacts, structural supply constraints, and global hash rate distribution.
        4. On-Chain & Whale Dynamics: Significant whale wallet movements and macro exchange inflows/outflows.
        5. Macro Liquidity Correlation: Indicators linking BTC price action to Fed rate cuts and global M2 supply growth.
        6. Ecosystem & TradFi Integration: Layer 2 TVL growth, Real-World Asset (RWA) tokenization, and major crypto industry M&A/IPOs.
        7. Hard Asset Benchmarks: Gold and Silver price action, ETF flows, and supply/demand dynamics for comparative store-of-value analysis.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="institutional money flows and on-chain facts",
            ignore_rule="speculative price predictions, technical analysis (support/resistance), or emotional market sentiment",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "etf",
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "inflow",
            "outflow",
            "whale",
            "treasury",
            "fasb",
            "halving",
            "hash rate",
            "tvl",
            "l2",
            "rwa",
            "tokenization",
            "gold",
            "silver",
            "saylor",
            "michael saylor",
            "fink",
            "larry fink",
            "gensler",
            "sec",
            "stablecoin",
            "defi",
            "staking",
            "solana",
            "sol",
            "sovereign fund",
            "strategic reserve",
            "crypto",
            "coinbase",
            "microstrategy",
            "mstr",
            "blackrock",
            "options",
            "liquidation",
            "on-chain",
            "tether",
            "usdt",
            "usdc",
            "miners",
            "mining",
            "supply squeeze",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    "Semiconductor": {
        "role": "Exact Text Extraction Algorithm: Semiconductor & Supply Chain",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Financials & CAPEX Cycles: Datacenter revenue, EPS/margins, and multi-year fab equipment spending (EUV/High-NA orders).
        2. Production & Packaging Bottlenecks: TSMC CoWoS/SOIC capacity, foundry utilization rates, Silicon Photonics, and Glass Substrate constraints.
        3. Next-Gen Roadmaps & Architecture: 2nm/1.4nm mass production timelines, Custom AI Silicon (ASIC) adoption, and chiplet/heterogeneous integration.
        4. Terafabs & Sovereign Infrastructure: Mega-scale AI compute clusters (e.g., xAI Colossus) and government-backed fab investments (CHIPS Act maturity).
        5. Leadership Forward Guidance: Strategic statements from key figures (Huang, Su, Musk, Gelsinger, Chang) regarding compute scaling and foundry expansion.
        6. Industry Consolidation & Capital Flows: Structural M&A, multi-billion dollar funding rounds, and IPOs across the semiconductor supply chain.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="semiconductor industry data",
            ignore_rule="speculative price predictions or technical analysis",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "revenue",
            "eps",
            "margin",
            "tsmc",
            "nvidia",
            "amd",
            "cowos",
            "foundry",
            "capex",
            "euv",
            "nm",
            "soic",
            "photonics",
            "chips act",
            "asic",
            "chiplet",
            "terafab",
            "musk",
            "elon musk",
            "jensen huang",
            "lisa su",
            "gelsinger",
            "intel",
            "samsung",
            "hbm",
            "high-na",
            "glass substrate",
            "ai chip",
            "gpu",
            "npu",
            "custom silicon",
            "datacenter",
            "server",
            "arm",
            "broadcom",
            "qualcomm",
            "micron",
            "wafer",
            "packaging",
            "asml",
            "yields",
            "supply chain",
            "fab",
            "texas instruments",
            "nxp",
            "memory",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    "AI": {
        "role": "Exact Text Extraction Algorithm: AI & Generative Models",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Compute & Edge Transition: Inference costs, hardware constraints (VRAM), and the structural shift from cloud training to Edge AI (AI PC/Smartphone supercycles).
        2. Enterprise ROI & Autonomous Agents: B2B deployment scale, API monetization, pilot-to-production conversion metrics, and the rollout of autonomous AI agents.
        3. Power Infrastructure Bottlenecks: Data center energy deals and physical power supply constraints (e.g., Nuclear/SMR contracts).
        4. Capital Flows & Consolidation: Startup valuations, multi-billion funding rounds, structural M&A, and AI IPOs.
        5. Sovereign AI Build-outs: Nation-state level investments and strategic domestic compute infrastructure.
        6. AGI & Frontier Capabilities: Structural leaps in multi-modal capabilities and timeline milestones for AGI development.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="AI industry data",
            ignore_rule="ethical debates, generic future outlooks, or technical analysis",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "compute",
            "vram",
            "gpu",
            "api",
            "b2b",
            "funding",
            "valuation",
            "nuclear",
            "smr",
            "roi",
            "agi",
            "multi-modal",
            "edge ai",
            "agent",
            "autonomous",
            "sam altman",
            "openai",
            "anthropic",
            "demis hassabis",
            "ilya sutskever",
            "xai",
            "grok",
            "llama",
            "claude",
            "gemini",
            "copilot",
            "chatgpt",
            "deepseek",
            "qwen",
            "moe",
            "generative ai",
            "llm",
            "large language model",
            "inference",
            "training",
            "ai agent",
            "data center",
            "datacenter",
            "power cooling",
            "liquid cooling",
            "open source",
            "parameters",
            "scaling law",
            "ipo",
            "m&a",
            "acquisition",
        ],
    },
    "Bio": {
        "role": "Exact Text Extraction Algorithm: Biotechnology & Pharmaceuticals",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Clinical & Regulatory Milestones: Phase 1/2/3 efficacy data (p-values) and FDA PDUFA decisions (Approvals/CRLs).
        2. Metabolic & GLP-1 Supercycle: Next-gen obesity pipelines (oral/muscle-preserving), expanding clinical indications, and CMO manufacturing constraints.
        3. AI-Driven Drug Discovery: Generative biology and AI models (e.g., AlphaFold) accelerating R&D timelines and reducing clinical failure rates.
        4. Advanced Therapies (CRISPR/Cell): Commercialization, scalability, and clinical milestones for gene-editing and cell therapies.
        5. Policy & Patent Cliffs: Drug pricing regulatory impacts (e.g., IRA) and revenue defense strategies against major blockbuster patent cliffs.
        6. Capital Flows & Consolidation: Multi-year pipeline depth, structural M&A, massive funding rounds, and BioTech IPOs.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="biotechnology industry data",
            ignore_rule="generic health advice or stock price predictions",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "clinical",
            "phase 1",
            "phase 2",
            "phase 3",
            "p-value",
            "fda",
            "pdufa",
            "crl",
            "cmo",
            "glp-1",
            "obesity",
            "patent",
            "ira",
            "m&a",
            "crispr",
            "alphafold",
            "eli lilly",
            "novo nordisk",
            "wegovy",
            "zepbound",
            "mrna",
            "car-t",
            "gene editing",
            "alzheimer",
            "oncology",
            "trial",
            "efficacy",
            "safety",
            "cro",
            "cdmo",
            "weight loss",
            "pipeline",
            "biotech",
            "pharma",
            "blockbuster",
            "fda approval",
            "immunotherapy",
            "biosimilar",
            "orphan drug",
            "ipo",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    "Aerospace": {
        "role": "Exact Text Extraction Algorithm: Aerospace & Space Economy",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Launch Economics & Reusability: Launch cost per ton, orbital frequency, payload capacity, and structural margin expansion from reusability.
        2. Commercial LEO & Connectivity: Satellite network metrics (e.g., Starlink subscribers/revenue) and Direct-to-Cell TAM expansion.
        3. Defense Budgets & Backlogs: Long-term DoD/NASA allocations, specific contract values, and defense contractor multi-year backlog growth.
        4. Next-Gen Defense Tech: DoD budget shifts toward autonomous AI defense (drone swarms) and hypersonic weapons systems.
        5. Lunar & Deep Space Economy: Contract awards and infrastructure milestones for lunar missions (e.g., Artemis).
        6. Capital Flows & Consolidation: Structural M&A, massive funding rounds, valuations, and IPOs across the space and defense sectors.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="aerospace industry data",
            ignore_rule="generic space exploration enthusiasm or stock price predictions",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "launch",
            "payload",
            "starlink",
            "nasa",
            "dod",
            "tam",
            "defense",
            "backlog",
            "orbit",
            "drone",
            "hypersonic",
            "artemis",
            "lunar",
            "spacex",
            "blue origin",
            "ula",
            "lockheed",
            "boeing",
            "northrop",
            "anduril",
            "palantir",
            "karp",
            "shield ai",
            "space",
            "satellite",
            "leo",
            "rocket",
            "missile",
            "rtx",
            "raytheon",
            "general dynamics",
            "defense budget",
            "pentagon",
            "procurement",
            "aviation",
            "aero",
            "military",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    "Software": {
        "role": "Exact Text Extraction Algorithm: Enterprise Software & Cloud Services",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Cloud Infrastructure & Pipeline: Hyperscaler (AWS, Azure, GCP) YoY revenue growth and multi-year Remaining Performance Obligations (RPO).
        2. AI Monetization & ARPU: Direct revenue impact from AI Copilot adoptions, enterprise ROI metrics, and structural Average Revenue Per User (ARPU) expansion.
        3. Pricing Model Paradigm Shift: Transition metrics tracking the shift from traditional seat-based SaaS to consumption and outcome-based 'AI Agent-as-a-Service' models.
        4. Ecosystem Lock-in & Retention: Net Retention Rate (NRR), Net Dollar Retention (NDR), and structural platform switching costs within AI-driven workflows.
        5. AI-Native Cybersecurity: Enterprise spending shifts toward AI model defense, automated SecOps, and vertical-specific security ecosystems.
        6. Industry Consolidation & Capital Flows: Enterprise software M&A, massive late-stage funding rounds, and high-profile AI/SaaS IPOs.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="software industry data",
            ignore_rule="generic feature updates or UI changes",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "aws",
            "azure",
            "google cloud",
            "gcp",
            "nrr",
            "ndr",
            "copilot",
            "arpu",
            "rpo",
            "saas",
            "cybersecurity",
            "nadella",
            "jassy",
            "pichai",
            "benioff",
            "salesforce",
            "servicenow",
            "crowdstrike",
            "palo alto",
            "databricks",
            "snowflake",
            "cloud computing",
            "enterprise",
            "b2b saas",
            "arr",
            "mrr",
            "churn",
            "adoption",
            "cisa",
            "zero trust",
            "endpoint security",
            "database",
            "oracle",
            "sap",
            "adobe",
            "workday",
            "palantir",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
        ],
    },
    "Others": {
        "role": "Exact Text Extraction Algorithm: Mega-Cap Tech & Consumer",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Hardware & Autonomous Scaling: Shipment volumes (e.g., EVs, smartphones), FSD/Robotaxi adoption metrics, and humanoid robotics milestones.
        2. Financials & Capital Returns: Free Cash Flow (FCF) generation, operating margin shifts, and FCF-driven massive share buybacks or dividend programs.
        3. Supply Chain Restructuring: "China+1" strategies, nearshoring/reshoring metrics, and mega-cap logistics cost reductions.
        4. Automation & Margin Expansion: Structural operating margin growth via physical robotics, automation, and enterprise AI integration.
        5. Corporate Actions & Anti-Trust: Major structural M&A, high-profile mega-cap IPOs, and critical regulatory anti-trust rulings.
        If a sentence contains any of these KPIs, extract it entirely. If the context requires the preceding or following sentence to make sense of the number, extract that surrounding block of text EXACTLY as written.""",
        "backstory": build_backstory(
            industry_focus="mega-cap and consumer industry data",
            ignore_rule="speculative price predictions, emotional sentiment, or CEO personal gossip",
        ),
        "task_description_template": build_task_template(),
        "keywords": [
            "delivery",
            "shipment",
            "fcf",
            "free cash flow",
            "margin",
            "logistics",
            "supply chain",
            "reshoring",
            "robotics",
            "automation",
            "buyback",
            "dividend",
            "robotaxi",
            "fsd",
            "tim cook",
            "zuckerberg",
            "bezos",
            "tesla",
            "apple",
            "meta",
            "amazon",
            "alphabet",
            "optimus",
            "figure ai",
            "consumer spending",
            "retail",
            "ecommerce",
            "ev",
            "electric vehicle",
            "iphone",
            "services revenue",
            "ad revenue",
            "advertising",
            "capex",
            "operating income",
            "cost cutting",
            "layoffs",
            "shareholder return",
            "walmart",
            "costco",
            "ipo",
            "m&a",
            "funding",
            "valuation",
            "acquisition",
            "anti-trust",
        ],
    },
}


def get_agent_config(category: str) -> dict:
    """
    Returns the specific prompt configuration for a given category.
    If the category isn't specifically defined in AGENT_CONFIGS,
    it falls back to a default exact-extraction template.
    """
    if category in AGENT_CONFIGS:
        return AGENT_CONFIGS[category]

    # Default fallback for other categories before they are hard-tuned
    return {
        "role": f"{category} Exact Text Extraction Algorithm",
        "goal": f"Your absolute goal is to scan the provided text and strictly COPY AND PASTE the exact original sentences that contain hard data or definitive statements regarding key performance indicators (KPIs) for {category}.",
        "backstory": build_backstory(industry_focus="industry data", ignore_rule=""),
        "task_description_template": build_task_template(),
    }
