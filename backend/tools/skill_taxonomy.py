"""
Skill taxonomy: canonical skill names, the category each belongs to, and the
surface forms ("aliases") that count as a mention of it.

Why aliases matter: JDs and resumes rarely spell a skill the same way. "k8s",
"sklearn", "LLMs", "YOLOv8", "RESTful APIs", "Postgres" all need to map to the
same canonical skill or the gap analysis reports false "missing" skills.

Each alias is a regex fragment matched case-insensitively with alphanumeric
boundaries on both sides (so "java" never matches inside "javascript", and
"rag" never matches inside "drag"). A few short, ambiguous names (R, Go, C,
Excel) are matched case-sensitively and only in list-like context — otherwise
"R&D", "go to market" and "excel at" would all register as skills.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...]
    # Regexes applied to the ORIGINAL (not lowercased) text, for ambiguous short names.
    case_sensitive: tuple[str, ...] = ()


# Lookbehind context for "this capitalized token is an item in a list of skills".
# Python lookbehinds must be fixed-width, hence one lookbehind per alternative.
_LIST_CTX = r"(?:^|(?<=, )|(?<=,)|(?<=\()|(?<=/)|(?<=: )|(?<=and )|(?<=or )|(?<=in )|(?<=with )|(?<=\| )|(?<=• ))"


def _listed(token: str, forbid_after: str = r"[A-Za-z0-9&+#'’-]") -> str:
    return _LIST_CTX + re.escape(token) + r"(?!" + forbid_after + r")"


_SKILLS: list[Skill] = [
    # --- Programming languages ---
    Skill("python", "languages", (r"python3?",)),
    Skill("java", "languages", (r"java",)),
    Skill("c++", "languages", (r"c\+\+", r"cpp")),
    Skill("c#", "languages", (r"c#", r"asp\.net", r"\.net", r"dotnet")),
    Skill("c", "languages", (), (_listed("C"),)),
    Skill("go", "languages", (r"golang",), (_listed("Go"),)),
    Skill("rust", "languages", (r"rust",)),
    Skill("javascript", "languages", (r"javascript", r"es6", r"ecmascript")),
    Skill("typescript", "languages", (r"typescript",)),
    Skill("sql", "languages", (r"sql", r"t-sql", r"pl/sql", r"mysql", r"postgres(?:ql)?", r"sqlite")),
    Skill("r", "languages", (), (_listed("R", forbid_after=r"[A-Za-z0-9&+#'’/-]"),)),
    Skill("scala", "languages", (r"scala",)),
    Skill("kotlin", "languages", (r"kotlin",)),
    Skill("swift", "languages", (r"swift(?:ui)?",)),
    Skill("matlab", "languages", (r"matlab",)),
    Skill("bash", "languages", (r"bash", r"shell scripting")),
    Skill("html/css", "languages", (r"html5?", r"css3?")),

    # --- Web & backend ---
    Skill("react", "web", (r"react(?:\.?js)?", r"react native")),
    Skill("next.js", "web", (r"next\.?js",)),
    Skill("vue", "web", (r"vue(?:\.?js)?",)),
    Skill("angular", "web", (r"angular(?:js)?",)),
    Skill("node.js", "web", (r"node\.?js", r"node js", r"express\.?js")),
    Skill("fastapi", "web", (r"fastapi",)),
    Skill("flask", "web", (r"flask",)),
    Skill("django", "web", (r"django",)),
    Skill("spring boot", "web", (r"spring ?boot", r"spring framework")),
    Skill("rest api", "web", (r"rest(?:ful)?[ -]?apis?", r"rest-?ful(?: web)?(?: services?)?", r"rest (?:services?|endpoints?)")),
    Skill("graphql", "web", (r"graphql",)),
    Skill("grpc", "web", (r"grpc",)),
    Skill("microservices", "web", (r"micro-?services?",)),
    Skill("streamlit", "web", (r"streamlit",)),
    Skill("tailwind", "web", (r"tailwind(?: ?css)?",)),

    # --- ML / AI ---
    Skill("machine learning", "ml_ai", (r"machine learning", r"ml")),
    Skill("deep learning", "ml_ai", (r"deep learning", r"neural networks?")),
    Skill("pytorch", "ml_ai", (r"pytorch", r"torch")),
    Skill("tensorflow", "ml_ai", (r"tensorflow", r"tf2")),
    Skill("keras", "ml_ai", (r"keras",)),
    Skill("jax", "ml_ai", (r"jax",)),
    Skill("scikit-learn", "ml_ai", (r"scikit[- ]?learn", r"sklearn")),
    Skill("xgboost", "ml_ai", (r"xgboost", r"lightgbm", r"catboost", r"gradient boost(?:ing|ed)")),
    Skill("huggingface", "ml_ai", (r"hugging ?face",)),
    Skill("transformers", "ml_ai", (r"transformers?(?: models?)?",)),
    Skill("llm", "ml_ai", (r"llms?", r"large language models?")),
    Skill("generative ai", "ml_ai", (r"generative ai", r"gen ?ai")),
    Skill("rag", "ml_ai", (r"rag", r"retrieval[- ]augmented generation"), (r"(?<![A-Z])RAG(?![A-Za-z])",)),  # also "HybridRAG"
    Skill("prompt engineering", "ml_ai", (r"prompt engineering", r"prompt design")),
    Skill("fine-tuning", "ml_ai", (r"fine[- ]?tun(?:e|ed|ing)", r"lora", r"peft")),
    Skill("langchain", "ml_ai", (r"langchain",)),
    Skill("langgraph", "ml_ai", (r"langgraph",)),
    Skill("llamaindex", "ml_ai", (r"llama ?index",)),
    Skill("crewai", "ml_ai", (r"crew ?ai",)),
    Skill("openai api", "ml_ai", (r"openai(?: api)?", r"gpt-?[345][a-z0-9.-]*")),
    Skill("ai agents", "ml_ai", (r"ai agents?", r"agentic", r"multi-agent")),
    Skill("faiss", "ml_ai", (r"faiss",)),
    Skill("pinecone", "ml_ai", (r"pinecone",)),
    Skill("vector database", "ml_ai", (r"vector (?:databases?|dbs?|stores?|search)", r"chroma(?:db)?", r"weaviate", r"qdrant", r"milvus", r"pgvector")),
    Skill("nlp", "ml_ai", (r"nlp", r"natural language processing")),
    Skill("computer vision", "ml_ai", (r"computer vision",)),
    Skill("opencv", "ml_ai", (r"opencv",)),
    Skill("mediapipe", "ml_ai", (r"mediapipe",)),
    Skill("yolo", "ml_ai", (r"yolo(?:v\d+)?",)),
    Skill("reinforcement learning", "ml_ai", (r"reinforcement learning",)),
    Skill("recommender systems", "ml_ai", (r"recommend(?:er|ation) (?:systems?|engines?)",)),
    Skill("time series", "ml_ai", (r"time[- ]series",)),
    Skill("mlops", "ml_ai", (r"mlops",)),
    Skill("mlflow", "ml_ai", (r"mlflow",)),
    Skill("model deployment", "ml_ai", (r"model deployment", r"deploy(?:ed|ing)? (?:ml |machine learning )?models?", r"model serving")),
    Skill("model monitoring", "ml_ai", (r"model monitoring",)),
    Skill("a/b testing", "ml_ai", (r"a/b test(?:s|ing)?", r"ab testing")),

    # --- Data ---
    Skill("pandas", "data", (r"pandas",)),
    Skill("numpy", "data", (r"numpy",)),
    Skill("statistics", "data", (r"statistics", r"statistical"),),
    Skill("linear algebra", "data", (r"linear algebra",)),
    Skill("data visualization", "data", (r"data visuali[sz]ation", r"matplotlib", r"seaborn", r"plotly", r"d3\.?js")),
    Skill("tableau", "data", (r"tableau",)),
    Skill("power bi", "data", (r"power ?bi",)),
    Skill("excel", "data", (r"microsoft excel", r"ms excel", r"spreadsheets?"), (_listed("Excel"),)),
    Skill("spark", "data", (r"(?:apache )?spark", r"pyspark")),
    Skill("pyspark", "data", (r"pyspark",)),
    Skill("hadoop", "data", (r"hadoop", r"hive", r"hdfs")),
    Skill("kafka", "data", (r"(?:apache )?kafka",)),
    Skill("airflow", "data", (r"(?:apache )?airflow",)),
    Skill("dbt", "data", (r"dbt",)),
    Skill("databricks", "data", (r"databricks",)),
    Skill("snowflake", "data", (r"snowflake",)),
    Skill("delta lake", "data", (r"delta lake",)),
    Skill("etl", "data", (r"etl", r"elt")),
    Skill("data pipeline", "data", (r"data pipelines?", r"data engineering")),

    # --- Databases ---
    Skill("postgresql", "databases", (r"postgres(?:ql)?",)),
    Skill("mysql", "databases", (r"mysql",)),
    Skill("mongodb", "databases", (r"mongo(?:db)?",)),
    Skill("redis", "databases", (r"redis",)),
    Skill("nosql", "databases", (r"nosql", r"dynamodb", r"cassandra")),
    Skill("firebase", "databases", (r"firebase", r"firestore")),

    # --- Cloud & DevOps ---
    Skill("aws", "cloud_devops", (r"aws", r"amazon web services", r"ec2", r"s3", r"sagemaker")),
    Skill("gcp", "cloud_devops", (r"gcp", r"google cloud(?: platform)?", r"bigquery", r"vertex ai")),
    Skill("azure", "cloud_devops", (r"azure",)),
    Skill("docker", "cloud_devops", (r"docker(?:file)?", r"containeri[sz](?:ed|ation)")),
    Skill("kubernetes", "cloud_devops", (r"kubernetes", r"k8s", r"eks", r"gke")),
    Skill("terraform", "cloud_devops", (r"terraform", r"infrastructure as code")),
    Skill("ci/cd", "cloud_devops", (r"ci\s*/\s*cd", r"continuous (?:integration|delivery|deployment)", r"jenkins", r"circleci")),
    Skill("github actions", "cloud_devops", (r"github actions",)),
    Skill("linux", "cloud_devops", (r"linux", r"unix")),

    # --- Engineering practices & tools ---
    Skill("git", "practices", (r"git", r"github", r"gitlab", r"version control")),
    Skill("unit testing", "practices", (r"unit test(?:s|ing)?", r"pytest", r"jest", r"junit", r"test-driven", r"tdd")),
    Skill("data structures", "practices", (r"data structures?",)),
    Skill("algorithms", "practices", (r"algorithms?",)),
    Skill("oop", "practices", (r"oop", r"object[- ]oriented(?: programming| design)?")),
    Skill("system design", "practices", (r"system design",)),
    Skill("distributed systems", "practices", (r"distributed systems?",)),
    Skill("agile", "practices", (r"agile", r"scrum", r"kanban")),
]

CATEGORY_LABELS = {
    "languages": "Languages",
    "web": "Web & Backend",
    "ml_ai": "ML / AI",
    "data": "Data",
    "databases": "Databases",
    "cloud_devops": "Cloud & DevOps",
    "practices": "Engineering Practices",
}

SKILLS: dict[str, Skill] = {s.name: s for s in _SKILLS}
SKILL_NAMES: list[str] = [s.name for s in _SKILLS]

_BOUNDARY_L = r"(?<![a-z0-9])"
_BOUNDARY_R = r"(?![a-z0-9])"

_COMPILED: dict[str, tuple[re.Pattern | None, list[re.Pattern]]] = {}
for _s in _SKILLS:
    ci = (
        re.compile(_BOUNDARY_L + "(?:" + "|".join(_s.aliases) + ")" + _BOUNDARY_R, re.IGNORECASE)
        if _s.aliases else None
    )
    cs = [re.compile(p, re.MULTILINE) for p in _s.case_sensitive]
    _COMPILED[_s.name] = (ci, cs)


def find_skill_spans(text: str) -> dict[str, list[tuple[int, int]]]:
    """Return {canonical_skill: [(start, end), ...]} for every skill mentioned in text."""
    found: dict[str, list[tuple[int, int]]] = {}
    for name, (ci, cs) in _COMPILED.items():
        spans: list[tuple[int, int]] = []
        if ci is not None:
            spans.extend(m.span() for m in ci.finditer(text))
        for pattern in cs:
            spans.extend(m.span() for m in pattern.finditer(text))
        if spans:
            found[name] = sorted(spans)
    return found


def find_skills(text: str) -> list[str]:
    """Canonical skills mentioned anywhere in text, sorted."""
    return sorted(find_skill_spans(text))


def canonicalize(raw_skill: str) -> list[str]:
    """
    Map a free-text skill (e.g. from the LLM: "PyTorch", "Experience with K8s")
    onto canonical taxonomy names. Returns [] if nothing in the taxonomy matches.
    """
    return find_skills(raw_skill)


def category_of(skill: str) -> str:
    s = SKILLS.get(skill)
    return s.category if s else "other"


def group_by_category(skills: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for skill in skills:
        groups.setdefault(category_of(skill), []).append(skill)
    return {cat: sorted(items) for cat, items in groups.items()}
