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
        1. Read the provided JSON data.
        2. Scan the content for the hard data KPIs defined in your Goal.
        3. Extract the exact sentences or context blocks.
        4. Output ONLY the raw extracted text blocks natively. Do not add any conversational filler, prefixes, numbering, or labels.
        5. CRITICAL: {critical_condition}
        
        Here are the articles to process:
        {{input_text}}"""


AGENT_CONFIGS = {
    "General": {
        "role": "Exact Text Extraction Algorithm: US Macroeconomics & Federal Reserve Policy",
        "goal": """Your absolute goal is to scan the provided text and strictly COPY AND PASTE the exact original sentences that contain hard data or definitive statements regarding the following Macroeconomic KPIs:
        1. Inflation metrics: CPI (Consumer Price Index), PCE, core inflation rates.
        2. Federal Reserve policy: Interest rate decisions (basis points), dot plot projections, quantitative tightening (QT) scale.
        3. Employment data: Non-Farm Payrolls (NFP), unemployment rate.
        4. Federal Reserve officials' direct quotes indicating hawkish or dovish policy shifts.
        5. Structural inflation trends (Core PCE trajectory) and Treasury yield curve shifts.
        6. M2 money supply, structural liquidity, and long-term Fed balance sheet (QT/QE) plans.
        7. Multi-year employment shifts and structural labor market changes.
        8. US debt issuance & Treasury yield curve shifts.
        9. AI-driven productivity metrics in GDP/employment.
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
        1. Spot ETF flows: Net inflows/outflows (specific dollar amounts, BTC volumes) for BTC/ETH ETFs.
        2. Institutional adoption: Corporate treasury purchases or official TradFi crypto service launches.
        3. On-chain data: Significant whale wallet movements or exchange inflows/outflows.
        4. Macro liquidity correlation: Statements linking BTC to Fed rate cuts or M2 supply growth.
        5. Corporate treasury adoption (e.g., FASB accounting changes, sovereign accumulation).
        6. Global hash rate distribution and structural supply dynamics (halving impacts over time).
        7. Layer 2 network TVL (Total Value Locked) growth and infrastructural expansion.
        8. Nation-state/Sovereign strategic reserve adoption.
        9. Real-World Asset (RWA) tokenization and TradFi integration on L2s.
        10. Institutional derivatives/options liquidity.
        11. Structural supply constraints.
        12. Gold price, Gold ETF flows, Gold mining stocks, Gold supply and demand dynamics.
        13. Silver price, Silver ETF flows, Silver mining stocks, Silver supply and demand dynamics.
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
        ],
    },
    "Semiconductor": {
        "role": "Exact Text Extraction Algorithm: Semiconductor & Supply Chain",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Revenue and Earnings: Datacenter revenue figures, EPS, and profit margins (Nvidia, AMD, TSMC).
        2. Production: TSMC CoWoS capacity, foundry utilization rates, and GPU shipment volumes.
        3. Capital Expenditures (CAPEX): Company spending on new fabs and equipment purchases.
        4. Multi-year CAPEX cycles (e.g., EUV/High-NA equipment orders).
        5. Next-gen node roadmaps (2nm/1.4nm development and mass production timelines).
        6. Structural advanced packaging (CoWoS/SOIC) capacity expansions.
        7. Silicon Photonics and next-gen advanced packaging (Glass Substrates/CoWoS) bottlenecks.
        8. Sovereign fab investments (CHIPS Act maturity).
        9. Custom AI Silicon (ASIC) market shifts.
        10. Chiplet architecture and heterogeneous integration.
        11. Mega-scale 'Terafab' clusters and infrastructure build-outs (e.g., Elon Musk's xAI Colossus or regional autonomous tech clusters).
        12. Statements from key figures (Elon Musk, Jensen Huang, Lisa Su, Pat Gelsinger, Morris Chang) regarding Terafabs, compute scaling, or future foundry investments.
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
        ],
    },
    "AI": {
        "role": "Exact Text Extraction Algorithm: AI & Generative Models",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Inference & Compute: Compute costs, latency metrics, and hardware requirements (VRAM usage).
        2. Enterprise Adoption: Number of API calls, enterprise B2B deployments, or specific monetization metrics.
        3. Funding & Valuation: Investment amounts, funding rounds for AI startups.
        4. Data center energy/power infrastructure deals (e.g., Nuclear/SMR contracts for AI data centers).
        5. Sovereign AI investments (nation-state level AI compute build-outs).
        6. Enterprise AI ROI (transition metrics from pilot programs to full production deployments).
        7. AGI development roadmaps and structural leaps in multi-modal capabilities.
        8. Transition from AI training to Edge AI/Inference (AI PC & Smartphone supercycle).
        9. Deployment of Autonomous B2B AI Agents.
        10. Physical power infrastructure bottlenecks (Nuclear/SMR contracts).
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
        ],
    },
    "Bio": {
        "role": "Exact Text Extraction Algorithm: Biotechnology & Pharmaceuticals",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Clinical Trial Data: Phase 1/2/3 statistical significance (e.g., weight-loss percentages), p-values.
        2. Regulatory Milestones: FDA PDUFA final decision dates, approvals, or Complete Response Letters (CRLs).
        3. Manufacturing & Supply: CMO (Contract Manufacturing) capacity constraints, specifically for next-generation GLP-1/obesity drugs (muscle-preserving or oral obesity drugs).
        4. Patent cliff timelines for major blockbuster drugs.
        5. Drug pricing policy impacts (e.g., Inflation Reduction Act - IRA effects).
        6. Expanding clinical indications for existing blockbuster drugs (e.g., GLP-1 applications beyond weight loss).
        7. Multi-year pipeline depth and structural M&A activities.
        8. Commercialization milestones of CRISPR/Gene-editing therapies.
        9. AI-driven drug discovery (AlphaFold) drastically cutting R&D timelines.
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
        ],
    },
    "Aerospace": {
        "role": "Exact Text Extraction Algorithm: Aerospace & Space Economy",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Launch Metrics: Launch cost per ton, orbital launch frequency, and payload capacity.
        2. Commercial Space: Starlink subscriber growth, revenue figures, and terminal deployments.
        3. Contract Awards: Specific NASA or DoD (Department of Defense) contract values and durations.
        4. Space economy TAM (Total Addressable Market) expansion, such as Direct-to-Cell satellite coverage.
        5. Reusability economics and structural margin expansion for launch vehicles.
        6. Long-term defense budget allocations and structural government contracts.
        7. Consolidation of defense contractors and multi-year backlog growth.
        8. Commercialization of Low Earth Orbit (Direct-to-Cell satellite networks).
        9. Autonomous AI defense tech (drone swarms/hypersonics) DoD budgets.
        10. Lunar economy (Artemis) contract awards.
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
        ],
    },
    "Software": {
        "role": "Exact Text Extraction Algorithm: Enterprise Software & Cloud Services",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Cloud Growth: AWS, Azure, or Google Cloud segment YoY growth and revenue figures.
        2. Customer Metrics: Net Retention Rate (NRR) and Net Dollar Retention (NDR).
        3. AI Monetization: Revenue impact from AI Copilot adoptions and ARPU (Average Revenue Per User) increases.
        4. Multi-year Remaining Performance Obligations (RPO) growth.
        5. Platform lock-in metrics and structural ecosystem switching costs.
        6. Transition metrics to AI-agentic workflows (shifts from seat-based pricing to consumption/outcome-based pricing).
        7. Shift from seat-based SaaS to consumption-based 'AI Agent-as-a-Service'.
        8. Cybersecurity spending for AI models, and vertical-specific AI platform ecosystem lock-in metrics.
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
        ],
    },
    "Others": {
        "role": "Exact Text Extraction Algorithm: Mega-Cap Tech & Consumer",
        "goal": """Your absolute goal is to strictly COPY AND PASTE the exact original sentences regarding:
        1. Hardware Delivery Units: Specific delivery/shipment numbers (e.g., Tesla EV deliveries, Apple iPhone shipments).
        2. Financial Health: Free Cash Flow (FCF) figures and operating margins.
        3. Retail & Logistics: Logistics cost reductions and supply chain bottlenecks for mega-cap retailers (e.g., Amazon).
        4. Supply chain restructuring (e.g., China+1 strategies, nearshoring, offshoring).
        5. Structural operating margin expansion via robotics, automation, and AI integration.
        6. Long-term capital return programs (massive share buybacks or structural dividend increases).
        7. Humanoid robotics and autonomous driving (Robotaxi) scaling metrics. FSD (Full Self-Driving) progress.
        8. US manufacturing reshoring/automation impacts on operating margins.
        9. Mega-cap FCF driven structural share buybacks.
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
