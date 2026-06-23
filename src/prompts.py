def build_backstory(industry_focus: str, ignore_rule: str) -> str:
    ignore_section = f"\n        - ALSO IGNORE {ignore_rule}." if ignore_rule else ""
    return f"""You are a deterministic, zero-creativity extraction engine for {industry_focus}.

        [CRITICAL GATING RULE]
        First, check the text's purpose. If the text is a personal finance tutorial, personal retirement calculator guide, credit card/retail mortgage shopping tip, budget-saving tip, or lifestyle column, you MUST immediately output exactly "NO_EXTRACTION" and terminate. Do not extract anything.

        STRICT RULES:
        - NEVER rewrite, summarize, or paraphrase any sentence. You are NOT an analyst or a summarizer.
        - Your output must be a strict, 100% exact substring match from the source text.{ignore_section}
        - NEVER write formatting tags (like <b>), markdown headers, or bullet-point reformats. Any output not in the source is an absolute failure.

        OUTPUT FORMAT:
        - Begin your response directly with the first extracted sentence. Do NOT prefix with "Output:", "Here is the text:", "This article...", "The following...", or any introductory/concluding remarks.
        - Do NOT describe or summarize what the article is about. Only copy-paste the exact original sentences that contain the KPIs defined in the Goal.
        - If no KPIs are found, output exactly "NO_EXTRACTION" with nothing else."""


def build_task_template(
    critical_condition: str = "If an article lacks the specific KPIs, you MUST output exactly the word: NO_EXTRACTION",
) -> str:
    return f"""
        1. CHRONOLOGICAL FLOW: Scan the text from the first to the last sentence. You MUST extract sentences in the exact order they appear in the source text. Never jump backward.
        2. KPI EXTRACTION: Identify and extract all "hard data KPIs" and "guidance figures" defined in the Goal. Extract the complete sentence to preserve context. Do NOT extract isolated numbers, fragmented phrases, or bullet points without context.
        3. NO TABLES/VISUALS: Absolutely do NOT extract markdown tables (pipes '|' or hyphens '-'), raw tabular lists, standalone chart labels, quote buttons, or graphic links. Ignore visual representations completely and only extract the surrounding explanatory narrative paragraphs.
        4. RAW TEXT OUTPUT: Output ONLY the raw extracted sentences natively.
           - Do NOT add any conversational prefixes, thoughts, or introduction wrappers (e.g., "For this article...").
           - **100% EXACT MATCH**: Every extracted sentence MUST be a strict, exact substring match from the original text.
        5. TITLE RELEVANCY: Only extract facts directly relevant to the main subject, entity, or topic described in the Title. Do NOT extract generic market summaries or boilerplate descriptions of other companies.
        6. NO DUPLICATES: Do NOT extract duplicate or virtually identical facts, even if they appear in slightly different formats (e.g., "$81B" vs "$81 billion"). Copy only the single best instance.
        7. CLUTTER FILTERING: Strictly ignore all promotional text, recommended reading lists, clickbait newsletters, or subscription notices at the end of the article.
        8. CRITICAL: {critical_condition}
        
        Here are the articles to process:
        {{input_text}}"""


# ------------------------------------------------------------------------------
# 10-Category Prompts & Configuration
# ------------------------------------------------------------------------------
AGENT_CONFIGS = {
    # 1) General (US Macroeconomics & Federal Reserve Policy)
    "General": {
        "role": "Exact Text Extraction Algorithm: US Macroeconomics, Federal Policy & Regulations",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Inflation Trajectories: CPI, PCE, Core PCE, and Supercore inflation month-over-month or YoY percentage changes.
        2. Fed Decisions & Projections: FOMC rate pause duration (at the 3.75% upper bound), dot plot voting distributions, and QT limits.
        3. Fiscal Actions & Liquidity: US fiscal deficit ratios, Treasury auction Bid-to-Cover demand ratios, and 2s10s yield curve inversion spreads.
        4. Forward Guidance: Direct quotes from Fed officials and hawkish/dovish semantic pivots.
        5. Structural Labor Shifts: Non-Farm Payrolls (NFP), unemployment rates, labor force participation, and wage growth percentages.
        6. Macro Housing Indicators: National average 30-year fixed mortgage rates, MBA mortgage application indexes, and Case-Shiller index growth. (DO NOT extract individual refinancing tutorials, home shopping guides, or mortgage calculators).
        7. Real Assets & Safe Havens: Gold and Silver spot prices, physical gold/silver ETF flows, and central bank gold accumulation metrics as indicators of fiat currency debasement.
        8. Macro Government Policies & Regulations: Trade tariffs, trade war metrics, antitrust/regulatory decisions affecting macro industries, and tax-deferred/tax-free account regulations (e.g., Roth IRA tax implications, BDC tax-free treatments).
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="US Macroeconomics, Federal Policy & Regulations",
            ignore_rule="speculative price predictions, technical analysis (support/resistance), or emotional market sentiment",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "cpi": 10,
            "pce": 10,
            "inflation": 5,
            "fed": 5,
            "federal reserve": 10,
            "interest rate": 5,
            "basis point": 5,
            "bps": 5,
            "dot plot": 10,
            "qt": 10,
            "nfp": 10,
            "payroll": 10,
            "unemployment": 10,
            "hawkish": 8,
            "dovish": 8,
            "yield curve": 10,
            "m2": 8,
            "treasury": 5,
            "macro": 5,
            "gdp": 10,
            "powell": 8,
            "yellen": 8,
            "recession": 5,
            "soft landing": 8,
            "cbdc": 8,
            "debt limit": 8,
            "fiscal": 5,
            "tariff": 10,
            "trade war": 10,
            "fomc": 10,
            "quantitative easing": 10,
            "qe": 10,
            "liquidity": 5,
            "rate cut": 10,
            "rate hike": 10,
            "core pce": 10,
            "treasury yield": 10,
            "pmi": 10,
            "ism": 10,
            "jobless claims": 10,
            "retail sales": 5,
            "consumer sentiment index": 10,
            "university of michigan consumer sentiment": 10,
            "wage growth": 8,
            "sofr": 10,
            "mortgage rate": 10,
            "mortgage applications": 10,
            "housing starts": 10,
            "home sales": 10,
            "case-shiller": 10,
            "delinquency": 5,
            "foreclosure": 8,
            "gold": 5,
            "silver": 5,
            "gld": 8,
            "slv": 8,
            "roth ira": 10,
            "ira": 5,
            "tax-free": 8,
            "tax-deferred": 8,
            "antitrust": 10,
            "anti-trust": 10,
            "regulation": 5,
            "policy": 5,
        },
    },
    # 2) Bitcoin (Bitcoin & Institutional Liquidity)
    "Bitcoin": {
        "role": "Exact Text Extraction Algorithm: Bitcoin & Institutional Liquidity",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. ETF Flow Dynamics: Daily/weekly net inflows or outflows for Spot BTC and ETH ETFs.
        2. Derivates & Options Volume: Options trading open interest and institutional volume for crypto ETFs.
        3. Corporate Treasury holdings: Public corporate treasury accumulation (e.g., MicroStrategy, Tesla) and premium/discount to Net Asset Value (NAV).
        4. Accounting Rules & Regs: FASB fair value accounting adoption metrics and SEC regulatory approval filings.
        5. Sovereign Strategic Reserves: Nation-state balance sheet purchases and sovereign wealth fund allocations.
        6. Supply & Fundamentals: Halving impacts, structural supply constraints, and global hash rate distribution.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="institutional money flows and on-chain facts",
            ignore_rule="speculative price predictions, technical analysis (support/resistance), or emotional market sentiment",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "bitcoin etf": 10,
            "btc etf": 10,
            "crypto etf": 10,
            "ethereum etf": 10,
            "eth etf": 10,
            "bitcoin": 8,
            "btc": 8,
            "ethereum": 8,
            "eth": 8,
            "inflow": 5,
            "outflow": 5,
            "whale": 5,
            "treasury": 3,
            "fasb": 10,
            "halving": 10,
            "hash rate": 10,
            "tvl": 10,
            "l2": 5,
            "rwa": 10,
            "tokenization": 8,
            "stablecoin": 10,
            "defi": 10,
            "staking": 8,
            "solana": 8,
            "sol": 8,
            "sovereign fund": 5,
            "strategic reserve": 8,
            "crypto": 8,
            "crypto options": 10,
            "crypto liquidation": 10,
            "on-chain": 10,
            "tether": 10,
            "usdt": 10,
            "usdc": 10,
            "miners": 5,
            "mining": 5,
            "supply squeeze": 5,
        },
    },
    # 3) Semiconductor (Semiconductor & Supply Chain)
    "Semiconductor": {
        "role": "Exact Text Extraction Algorithm: Semiconductor & Supply Chain",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Foundry CapEx & Revenue: TSMC, Samsung, and Intel datacenter/foundry revenue growth rates and capital expenditures.
        2. Lithography Orders: EUV and High-NA EUV system shipment numbers, average selling prices (ASP), and backlog orders.
        3. Advanced Packaging Milestones: CoWoS/SOIC monthly wafer capacity metrics, Glass Substrate yields, and Silicon Photonics adoption percentages.
        4. High Bandwidth Memory: HBM3e/HBM4 product qualifications, shipment allocation percentages, and pricing premiums.
        5. CSP Custom ASICs: In-house ASIC chip tape-out schedules and production volume versus NVIDIA/AMD GPU shipments.
        6. Industry Consolidation & Flows: Semiconductor M&A valuations, public IPO offerings, and venture capital deals.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="semiconductor industry data",
            ignore_rule="speculative price predictions or technical analysis",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "cowos": 10,
            "foundry": 8,
            "euv": 10,
            "nm": 8,
            "soic": 10,
            "photonics": 10,
            "chips act": 10,
            "asic": 10,
            "chiplet": 10,
            "terafab": 10,
            "hbm": 10,
            "high-na": 10,
            "glass substrate": 10,
            "ai chip": 10,
            "gpu": 8,
            "npu": 8,
            "custom silicon": 10,
            "datacenter": 2,
            "server": 2,
            "wafer": 10,
            "packaging": 8,
            "yields": 5,
            "supply chain": 2,
            "fab": 5,
            "memory": 3,
            "nvidia": 10,
            "nvda": 10,
            "broadcom": 10,
            "avgo": 10,
            "intel": 10,
            "intc": 10,
            "amd": 10,
            "tsmc": 10,
            "tsm": 10,
            "qualcomm": 10,
            "qcom": 10,
            "samsung": 10,
            "sk hynix": 10,
            "hynix": 10,
            "micron": 10,
            "asml": 10,
        },
    },
    # 4) AI/Robotics/EV (Artificial Intelligence, Robotics & Electric Vehicles)
    "AI/Robotics/EV": {
        "role": "Exact Text Extraction Algorithm: AI, Robotics & EV",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. AI Model Scales & Costs: Training parameters, dataset token sizes, training run costs, and API pricing per million tokens.
        2. Enterprise ROI & Edge AI Specs: Corporate agent deployment conversion rates, productivity gains, and local execution hardware requirements (e.g., NPU TOPS).
        3. Autonomous Systems & Fleet Operations: Cumulative FSD (Full Self-Driving) miles driven, average miles between disengagements, and operating Robotaxi vehicle counts.
        4. Humanoid & Industrial Robotics: Factory floor humanoid deployments, unit manufacturing costs, and logistics/warehousing automation order backlogs.
        5. EV & Advanced Mobility Hardware: EV production volume and deliveries, battery cell costs ($/kWh), solid-state or next-gen battery development, and advanced hybrid/electric vehicle platform architectures.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="AI, robotics, autonomous systems, and advanced electric mobility hardware",
            ignore_rule="ethical debates, generic future outlooks, speculative consumer product release hype, retail auto sales dealer promotions, or personal stock tips",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "compute": 5,
            "vram": 8,
            "gpu": 8,
            "api": 2,
            "b2b": 2,
            "roi": 2,
            "agi": 10,
            "multi-modal": 8,
            "edge ai": 10,
            "agent": 5,
            "autonomous": 3,
            "grok": 10,
            "llama": 10,
            "claude": 10,
            "gemini": 10,
            "copilot": 8,
            "chatgpt": 10,
            "deepseek": 10,
            "qwen": 10,
            "moe": 8,
            "generative ai": 10,
            "llm": 10,
            "large language model": 10,
            "inference": 8,
            "training": 5,
            "ai agent": 10,
            "data center": 2,
            "datacenter": 2,
            "power cooling": 5,
            "liquid cooling": 8,
            "open source": 2,
            "parameters": 5,
            "scaling law": 8,
            "openai": 10,
            "anthropic": 10,
            "xai": 10,
            "suno": 8,
            "runway": 8,
            "palantir": 10,
            "pltr": 10,
            "fsd": 10,
            "full self-driving": 10,
            "robotaxi": 10,
            "humanoid": 10,
            "humanoid robot": 10,
            "optimus": 10,
            "figure ai": 10,
            "agv": 10,
            "automated guided vehicle": 10,
            "warehouse automation": 8,
            "industrial automation": 8,
            "battery cell": 8,
            "solid-state battery": 10,
            "robot": 8,
            "robotics": 8,
            "autonomy": 5,
            "autonomous driving": 10,
            "disengagement": 10,
            "tesla": 10,
            "tsla": 10,
            "ev": 10,
            "electric vehicle": 10,
            "autonomous vehicle": 10,
            "self-driving": 10,
            "lithium": 5,
        },
    },
    # 5) Power/Grid (Power Infrastructure & Energy)
    "Power/Grid": {
        "role": "Exact Text Extraction Algorithm: Power Infrastructure & Energy",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Power Purchase Agreements: Megawatts (MW) or Gigawatts (GW) of energy secured under contract by tech companies, and contract pricing per megawatt-hour ($/MWh).
        2. Clean Energy & Nuclear Capacity: Nuclear plant power allocations, Small Modular Reactor (SMR) development schedules, and uranium fuel supply metrics.
        3. Grid Hardware lead times: Transformer lead times (months/weeks), switchgear backlogs, and utility substation infrastructure CapEx.
        4. Utility-Scale Battery Storage: Energy Storage System (ESS) capacity installations (MWh), and battery storage discharge duration metrics.
        5. Utility Infrastructure Consolidation: Energy sector M&A deals, utility IPOs, and capital flows for power grid upgrades.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="power grid, utilities, and energy infrastructure data",
            ignore_rule="general climate changes or retail energy bills",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "power purchase agreement": 10,
            "ppa": 8,
            "nuclear": 8,
            "smr": 10,
            "uranium": 10,
            "transformer": 10,
            "switchgear": 10,
            "substation": 10,
            "ess": 8,
            "energy storage": 8,
            "battery storage": 8,
            "utility": 5,
            "utilities": 5,
            "clean energy": 8,
            "electricity": 5,
            "power grid": 10,
            "grid interconnect": 10,
            "megawatts": 8,
            "gigawatts": 8,
            "mwh": 8,
            "gwh": 8,
        },
    },
    # 6) Software (Enterprise Software & Cloud Services)
    "Software": {
        "role": "Exact Text Extraction Algorithm: Enterprise Software & Cloud Services",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Consumption Monetization: Revenue percentages from consumption-based or outcome-based billing vs. traditional seat-based licensing.
        2. Remaining Performance Obligations: Total RPO contract values, NRR (Net Retention Rate) percentages, and NDR (Net Dollar Retention) rates.
        3. SecOps Automation Budgets: Cloud security automation spend, platform consolidation numbers, and active threat detection rates.
        4. SaaS Churn & Adoption: Enterprise software user seat churn rates, agentic SaaS seat replacement metrics, and software seat pricing changes.
        5. Enterprise Software M&A/IPOs: Software company mergers, valuation multiples of acquisitions, and SaaS IPO pipeline status.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="software industry data",
            ignore_rule="generic feature updates or UI changes",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "aws": 10,
            "azure": 10,
            "google cloud": 10,
            "gcp": 10,
            "nrr": 10,
            "ndr": 10,
            "copilot": 5,
            "arpu": 8,
            "rpo": 10,
            "saas": 10,
            "cybersecurity": 8,
            "cloud computing": 8,
            "enterprise": 2,
            "b2b saas": 10,
            "arr": 10,
            "mrr": 10,
            "churn": 5,
            "adoption": 2,
            "cisa": 10,
            "zero trust": 10,
            "endpoint security": 10,
            "database": 5,
            "google": 10,
            "alphabet": 10,
            "googl": 10,
            "goog": 10,
            "meta": 10,
            "facebook": 10,
            "amazon": 10,
            "amzn": 10,
            "apple": 10,
            "aapl": 10,
            "microsoft": 10,
            "msft": 10,
            "oracle": 10,
            "orcl": 10,
            "salesforce": 10,
            "crm": 10,
            "service now": 10,
            "now": 8,
            "crowdstrike": 10,
            "crwd": 10,
            "snowflake": 10,
            "snow": 10,
        },
    },
    # 7) Aerospace (Aerospace, Space Economy & Defense)
    "Aerospace": {
        "role": "Exact Text Extraction Algorithm: Aerospace & Space Economy",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Launch Cost & Freq: Cost per launch payload kilogram ($/kg), rocket booster reuse cycles, and annual/monthly launch counts.
        2. Satellite Broadband (Starlink): Active subscribers, monthly average revenue per user (ARPU), and total satellite constellations.
        3. Autonomous Defense Systems: DoD contract award values for AI-driven drone swarms, tactical hypersonic speed metrics, and hardware quantities.
        4. Military Software Backlogs: Palantir and Anduril multi-year contract values and government defense budget allocations.
        5. Commercial aviation backlogs: Boeing and Airbus commercial jet order backlog years, monthly production rates, and delivery delays.
        6. Defense Consolidation: Aerospace & defense mergers & acquisitions (M&A), defense contractor IPOs, and venture funding for defense tech startups.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="aerospace industry data",
            ignore_rule="generic space exploration enthusiasm or stock price predictions",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "launch": 5,
            "payload": 8,
            "starlink": 10,
            "nasa": 8,
            "dod": 8,
            "tam": 2,
            "defense": 5,
            "backlog": 5,
            "orbit": 8,
            "drone": 8,
            "hypersonic": 10,
            "artemis": 10,
            "lunar": 8,
            "space": 5,
            "satellite": 8,
            "leo": 8,
            "rocket": 8,
            "missile": 8,
            "defense budget": 10,
            "pentagon": 8,
            "procurement": 5,
            "aviation": 5,
            "aero": 5,
            "military": 5,
            "spacex": 10,
            "blue origin": 10,
            "boeing": 10,
            "ba": 10,
            "airbus": 10,
            "anduril": 10,
            "defense contractor": 8,
        },
    },
    # 8) Bio (Biotechnology & Pharmaceuticals)
    "Bio": {
        "role": "Exact Text Extraction Algorithm: Biotechnology & Pharmaceuticals",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Trial Efficacy & Stats: Phase 1/2/3 clinical trial primary endpoint efficacy percentages, patient weight loss percentages, and p-values.
        2. Regulatory Decisions: FDA PDUFA action dates, CRL (Complete Response Letter) rejections, and European EMA approvals.
        3. GLP-1 Indication Expansions: Cardiovascular event risk reductions, kidney disease delay metrics, and sleep apnea trial data.
        4. Obesity pipeline formulations: Oral GLP-1 bioavailability percentages, muscle-mass preservation ratios, and dosing schedules.
        5. CDMO capacity expansions: Vial filling line throughput capacities, CDMO contract backlogs, and syringe supply metrics.
        6. BioTech Consolidation: Biotechnology mergers & acquisitions (M&A), major VC funding rounds, and clinical-stage BioTech IPOs.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="biotechnology industry data",
            ignore_rule="generic health advice or stock price predictions",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "clinical": 5,
            "phase 1": 10,
            "phase 2": 10,
            "phase 3": 10,
            "p-value": 10,
            "fda": 8,
            "pdufa": 10,
            "crl": 10,
            "cmo": 8,
            "glp-1": 10,
            "obesity": 8,
            "patent": 5,
            "ira": 5,
            "crispr": 10,
            "alphafold": 10,
            "wegovy": 10,
            "zepbound": 10,
            "mrna": 10,
            "car-t": 10,
            "gene editing": 10,
            "alzheimer": 8,
            "oncology": 8,
            "trial": 5,
            "efficacy": 8,
            "safety": 5,
            "cro": 8,
            "cdmo": 10,
            "weight loss": 8,
            "pipeline": 5,
            "biotech": 8,
            "pharma": 8,
            "blockbuster": 8,
            "fda approval": 10,
            "immunotherapy": 10,
            "biosimilar": 8,
            "orphan drug": 8,
        },
    },
    # 9) Consumer/Retail (Consumer Spending, E-Commerce & Retail)
    "Consumer/Retail": {
        "role": "Exact Text Extraction Algorithm: Consumer Spending, E-Commerce & Retail",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. E-Commerce & Retail Sales: Same-store sales (SSS) growth, comparable store sales (comps), and retail operating margins for major retail brands.
        2. Consumer Spending Trends: Direct metrics on consumer discretionary spend, household balance sheet changes, and metropolitan delivery expansion.
        3. E-Commerce platform activity: GMV (Gross Merchandise Value) growth, merchant subscription revenues, and active customer acquisition costs.
        4. Logistics & Supply Chain: Metropolitan fulfillment center capacity, same-day delivery penetration, and inventory turn ratios.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="consumer, retail, and e-commerce industry data",
            ignore_rule="retail product promotional discounts, coupon codes, individual consumer shopping reviews, or personal stock tips",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "walmart": 10,
            "wmt": 10,
            "costco": 10,
            "cost": 10,
            "target": 10,
            "tgt": 10,
            "lulu": 10,
            "lululemon": 10,
            "nike": 10,
            "nke": 10,
            "mcdonald's": 10,
            "mcd": 10,
            "coca-cola": 10,
            "coca cola": 10,
            "ko": 10,
            "starbucks": 10,
            "sbux": 10,
            "macy's": 10,
            "macy": 10,
            "retail": 10,
            "retailer": 10,
            "consumer spending": 10,
            "same-store sales": 10,
            "comps": 8,
            "ecommerce": 8,
            "shopify": 10,
            "shop": 8,
            "consumer sentiment": 5,
            "brand": 5,
            "apparel": 8,
            "restaurant": 8,
            "grocery": 8,
        },
    },
    # 10) Others (Miscellaneous Industries & General Business Dynamics)
    "Others": {
        "role": "Exact Text Extraction Algorithm: Miscellaneous Industries & General Business Dynamics",
        "goal": """Your absolute goal is to scan the text and strictly COPY AND PASTE the exact original sentences containing institutional-level hard data or definitive statements regarding:
        1. Miscellaneous Industry Performance: Traditional manufacturing, real estate (non-data center), food/beverage retail (excluding mega brands), and materials.
        2. General Corporate Actions: Multi-industry business announcements, generic cost-cutting, general board/executive changes, and general shareholder distributions not mapped elsewhere.
        If a sentence contains any of these KPIs, extract it entirely.""",
        "backstory": build_backstory(
            industry_focus="general corporate and miscellaneous industry data",
            ignore_rule="speculative price predictions, emotional market sentiment, or CEO personal gossip",
        ),
        "task_description_template": build_task_template(),
        "keywords": {
            "delivery": 1,
            "shipment": 1,
            "margin": 1,
            "revenue": 1,
            "eps": 1,
            "earnings": 1,
            "net income": 1,
            "operating income": 1,
            "gross margin": 1,
            "operating margin": 1,
            "reshoring": 5,
            "china+1": 5,
            "nearshoring": 5,
            "layoffs": 5,
            "cost cutting": 5,
            "buyback": 2,
            "dividend": 2,
            "shareholder return": 2,
        },
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
