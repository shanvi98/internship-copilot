"""
Tool 5: ATS Keyword Analyzer
The skill taxonomy only knows about tech skills. Applicant tracking systems
match on *any* term the JD repeats — domain words ("fraud", "payments",
"healthcare"), methods ("forecasting", "experimentation"), artifacts
("dashboards", "APIs"). This tool pulls the JD's most prominent terms
(single words and two-word phrases), and checks which ones the resume uses.

Pure Python, no dependencies, instant.
"""

import re
from collections import Counter

_STOPWORDS = set("""
a about above across after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each either etc few for from further had has have
having he her here hers him his how i if in into is it its itself just let me more most my no nor not now of off
on once only or other our ours out over own per same she should so some such than that the their theirs them then
there these they this those through to too under until up upon us very via was we were what when where which while
who whom why will with within without would you your yours yourself
able ability across already always among another anyone anything around become best better big bring build
building built candidate candidates company day days do eg e.g ie i.e enjoy equal etc every excellent experience
experiences familiar familiarity get go goal goals good great help helping highly ideal include includes including
individual job join key knowledge large learn learning level like looking make making many may must need needs new
nice opportunity opportunities part people plus position preferred program qualifications related required
requirement requirements responsibilities role skills strong summer team teams time type understanding use used
using want way well wide work working world year years you'll we're you're we'll it's they're
intern interns internship internships summer fall spring winter salary benefits perks pay paid range hour hours
week weeks month months apply applicants applicant please location remote hybrid onsite on-site office full-time
part-time employer employees employee free lunch offer offers starting start date dates
""".split())

# Sections that describe the company or compensation rather than the work; their
# words aren't what an ATS screens the resume against.
_BOILERPLATE_HEADING = re.compile(
    r"^\W*(benefits|perks|compensation|salary|pay range|what we offer|about us|about the company|"
    r"who we are|equal (?:employment )?opportunity|eeo|accommodations?|privacy)\b",
    re.IGNORECASE,
)
# Headings that start a section about the work itself, ending a boilerplate section.
_WORK_HEADING = re.compile(
    r"^\W*(requirements?|qualifications|responsibilities|what you(?:'|’)?ll (?:do|need)|"
    r"what we(?:'|’)?re looking for|nice[- ]to[- ]haves?|preferred|bonus|you have|skills|the role|about the role)\b",
    re.IGNORECASE,
)


def _strip_boilerplate(text: str) -> str:
    kept, skipping = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if _BOILERPLATE_HEADING.match(stripped):
            # "Benefits: free lunch" is an inline section — drop just this line;
            # a bare "Benefits" heading drops everything until the next work heading.
            skipping = not re.search(r":\s*\S", stripped)
            continue
        if skipping and (stripped.endswith(":") or _WORK_HEADING.match(stripped)):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


_TOKEN = re.compile(r"[a-z][a-z0-9+#.\-/]*[a-z0-9+#]|[a-z]")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower())]


def _stem(word: str) -> str:
    """Crude plural/verb-form folding so 'dashboards' matches 'dashboard', 'modeling' ~ 'model'."""
    for suffix in ("ies", "ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            base = word[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return word


def _terms(text: str) -> Counter:
    tokens = _tokens(text)
    counts: Counter = Counter()
    for i, tok in enumerate(tokens):
        if tok in _STOPWORDS or len(tok) < 3 or tok.isdigit():
            continue
        counts[tok] += 1
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in _STOPWORDS and len(nxt) >= 3 and not nxt.isdigit():
                counts[f"{tok} {nxt}"] += 1
    return counts


def analyze_keywords(resume_text: str, jd_text: str, top_n: int = 25) -> dict:
    """
    Returns the JD's top terms (ranked by frequency, two-word phrases weighted up)
    and whether each appears in the resume.
    """
    jd_counts = _terms(_strip_boilerplate(jd_text))
    # A phrase repeated in the JD is a much stronger signal than either word alone.
    scored = {t: c * (1.6 if " " in t else 1.0) for t, c in jd_counts.items() if " " not in t or c >= 2}
    ranked = sorted(scored, key=lambda t: (-scored[t], t))

    chosen: list[str] = []
    for term in ranked:
        # skip single words already covered by a chosen phrase, and vice-versa duplicates
        if any(term in c.split() or c in term.split() for c in chosen if " " in c or " " in term):
            continue
        chosen.append(term)
        if len(chosen) >= top_n:
            break

    resume_stems = {_stem(t) for t in _tokens(resume_text)}
    resume_lower = " ".join(_tokens(resume_text))

    keywords = []
    for term in chosen:
        if " " in term:
            found = term in resume_lower or all(_stem(w) in resume_stems for w in term.split())
        else:
            found = _stem(term) in resume_stems
        keywords.append({"term": term, "jd_count": jd_counts[term], "in_resume": found})

    present = sum(1 for k in keywords if k["in_resume"])
    return {
        "keywords": keywords,
        "missing_keywords": [k["term"] for k in keywords if not k["in_resume"]],
        "keyword_match_pct": round(present / len(keywords) * 100, 1) if keywords else 0.0,
    }


if __name__ == "__main__":
    jd = """Data Science Intern. You will build forecasting models and dashboards for our payments
    team, run experiments, and present insights to stakeholders. Forecasting experience and
    dashboard tools (Tableau) preferred. Payments or fraud domain knowledge is a plus."""
    resume = "Built a dashboard in Tableau tracking sales; created forecasting model for demand."
    import json

    print(json.dumps(analyze_keywords(resume, jd, top_n=10), indent=2))
