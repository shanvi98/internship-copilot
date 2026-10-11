"""
Fetch a job description from a posting URL.

Strategy: many ATS-hosted job pages (Greenhouse, Lever, Ashby, Workday, most
company career sites) embed a schema.org `JobPosting` JSON-LD block for Google
Jobs — that's the cleanest source of the title and description. If there isn't
one, fall back to the page's visible text with nav/script/style stripped.

Security: fetching arbitrary URLs from a server is an SSRF risk, so only
http(s) URLs that resolve to public IP addresses are allowed, redirects are
followed manually (each hop re-validated), and response size is capped.
Set ALLOW_URL_FETCH=0 to disable the feature entirely on a public deployment.
"""

import html
import ipaddress
import json
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 4
TIMEOUT_SECONDS = 10
_USER_AGENT = "Mozilla/5.0 (compatible; InternshipCopilot/0.2; +https://github.com/shanvi98/internship-copilot)"


class FetchError(ValueError):
    pass


def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError("Only http(s) URLs are supported")
    if not parsed.hostname:
        raise FetchError("URL has no host")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise FetchError(f"Couldn't resolve host '{parsed.hostname}'") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise FetchError("URL points to a private or reserved network address")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form", "button"}
    _BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self._in_jsonld = False
        self.jsonld: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type", "").lower() == "application/ld+json":
            self._in_jsonld = True
            return
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("• ")

    def handle_endtag(self, tag):
        if self._in_jsonld and tag == "script":
            self._in_jsonld = False
            return
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_jsonld:
            self.jsonld.append(data)
        elif self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)


def _html_to_text(fragment: str) -> str:
    parser = _TextExtractor()
    parser.feed(fragment)
    return _normalize("".join(parser.parts))


def _normalize(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_job_posting(blob) -> dict | None:
    if isinstance(blob, list):
        for item in blob:
            found = _find_job_posting(item)
            if found:
                return found
    elif isinstance(blob, dict):
        kind = blob.get("@type")
        if kind == "JobPosting" or (isinstance(kind, list) and "JobPosting" in kind):
            return blob
        for key in ("@graph", "mainEntity"):
            if key in blob:
                found = _find_job_posting(blob[key])
                if found:
                    return found
    return None


def parse_job_page(page_html: str) -> dict:
    parser = _TextExtractor()
    parser.feed(page_html)

    for raw in parser.jsonld:
        try:
            posting = _find_job_posting(json.loads(raw))
        except json.JSONDecodeError:
            continue
        if posting and posting.get("description"):
            org = posting.get("hiringOrganization") or {}
            company = org.get("name", "") if isinstance(org, dict) else ""
            title = posting.get("title", "")
            description = posting["description"]
            if "&lt;" in description:  # many sites entity-escape the HTML inside JSON-LD
                description = html.unescape(description)
            body = _html_to_text(description)
            text = f"{title}\n{company}\n\n{body}".strip() if title else body
            return {"title": title, "company": company, "text": text, "source": "json-ld"}

    return {
        "title": _normalize(parser.title),
        "company": "",
        "text": _normalize("".join(parser.parts)),
        "source": "page-text",
    }


def fetch_job_description(url: str) -> dict:
    import requests

    url = url.strip()
    for _ in range(MAX_REDIRECTS + 1):
        _assert_public_url(url)
        try:
            resp = requests.get(
                url, timeout=TIMEOUT_SECONDS, allow_redirects=False, stream=True,
                headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            )
        except requests.RequestException as e:
            raise FetchError(f"Couldn't fetch the page: {e.__class__.__name__}") from e
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise FetchError("Redirect without a Location header")
            url = urljoin(url, location)
            continue
        break
    else:
        raise FetchError("Too many redirects")

    with resp:
        if resp.status_code >= 400:
            raise FetchError(f"The page returned HTTP {resp.status_code}")
        content_type = resp.headers.get("Content-Type", "")
        if content_type and "html" not in content_type and "text" not in content_type:
            raise FetchError(f"Not a web page (Content-Type: {content_type.split(';')[0]})")
        body = b""
        for chunk in resp.iter_content(64 * 1024):
            body += chunk
            if len(body) > MAX_BYTES:
                raise FetchError("Page is too large")
        page = body.decode(resp.encoding or "utf-8", errors="replace")

    result = parse_job_page(page)
    if len(result["text"].split()) < 30:
        raise FetchError(
            "Couldn't find a job description on that page — it may be rendered by JavaScript. "
            "Paste the text instead."
        )
    result["url"] = url
    return result
