import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

app = FastAPI()

# ── Config ──────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PB_NEWS_URL = os.environ.get("PB_NEWS_URL", "https://pb-news.croquetwade.com")
PB_NEWS_EMAIL = os.environ.get(
    "PB_NEWS_ADMIN_EMAIL", os.environ.get("PB_ADMIN_EMAIL", "")
)
PB_NEWS_PASSWORD = os.environ.get(
    "PB_NEWS_ADMIN_PASSWORD", os.environ.get("PB_ADMIN_PASSWORD", "")
)
POLISH_MODEL = "deepseek/deepseek-v3.2"

# ── PocketBase auth (copied from newsroom-share/app.py) ────────────
_pb_token: str = ""


def _auth() -> str:
    r = http_requests.post(
        f"{PB_NEWS_URL}/api/collections/_superusers/auth-with-password",
        json={"identity": PB_NEWS_EMAIL, "password": PB_NEWS_PASSWORD},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["token"]


def get_token() -> str:
    global _pb_token
    if not _pb_token:
        _pb_token = _auth()
    return _pb_token


def refresh_token() -> str:
    global _pb_token
    _pb_token = _auth()
    return _pb_token


# ── Helpers ─────────────────────────────────────────────────────────
def _pb_request(method: str, path: str, **kwargs):
    """Make a PB request with 401-retry."""
    url = f"{PB_NEWS_URL}{path}"
    token = get_token()
    r = getattr(http_requests, method)(
        url, headers={"Authorization": f"Bearer {token}"}, timeout=15, **kwargs
    )
    if r.status_code in (401, 403):
        token = refresh_token()
        r = getattr(http_requests, method)(
            url, headers={"Authorization": f"Bearer {token}"}, timeout=15, **kwargs
        )
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r


# ── Pydantic models ────────────────────────────────────────────────
class PatchBody(BaseModel):
    title: str | None = None
    body: str | None = None
    excerpt: str | None = None
    category: str | None = None


class RejectBody(BaseModel):
    feedback: str = ""


class CreateBody(BaseModel):
    title: str
    body: str = ""
    excerpt: str = ""
    category: str = "News"


# ── Routes ──────────────────────────────────────────────────────────
@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/articles/{article_id}/cover")
async def cover_image(
    article_id: str,
    thumb: str = "200x140",
    filename: str | None = None,
    collection_id: str | None = None,
):
    """Redirect browser directly to PocketBase file URL (files are public — no proxy needed)."""
    if not filename or not collection_id:
        article = _pb_request(
            "get", f"/api/collections/news_articles/records/{article_id}"
        ).json()
        filename = article.get("cover_image")
        collection_id = article.get("collectionId", "")
    if not filename:
        raise HTTPException(status_code=404, detail="No cover image")
    pb_url = (
        f"{PB_NEWS_URL}/api/files/{collection_id}/{article_id}/{filename}"
        f"?thumb={thumb}"
    )
    return RedirectResponse(url=pb_url, status_code=302)


@app.get("/api/articles")
async def list_articles(
    status: str | None = None,
    search: str | None = None,
    page: int = 1,
    perPage: int = 20,
):
    filter_parts = ["status!='archived'"]
    if status:
        filter_parts.append(f"status='{status}'")
    if search:
        safe = search.replace("'", "")
        filter_parts.append(
            f"(title~'{safe}' || body~'{safe}' || author_name~'{safe}')"
        )
    pb_filter = " && ".join(filter_parts)
    r = _pb_request(
        "get",
        "/api/collections/news_articles/records",
        params={"sort": "-updated", "filter": pb_filter, "page": page, "perPage": perPage},
    )
    return r.json()


@app.get("/api/articles/{article_id}")
async def get_article(article_id: str):
    r = _pb_request("get", f"/api/collections/news_articles/records/{article_id}")
    return r.json()


@app.patch("/api/articles/{article_id}")
async def patch_article(article_id: str, body: PatchBody):
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=422, detail="No fields to update")
    r = _pb_request(
        "patch", f"/api/collections/news_articles/records/{article_id}", json=data
    )
    return r.json()


@app.post("/api/articles/{article_id}/publish")
async def publish_article(article_id: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000Z")
    r = _pb_request(
        "patch",
        f"/api/collections/news_articles/records/{article_id}",
        json={"status": "published", "published_at": now, "review_feedback": ""},
    )
    return r.json()


@app.post("/api/articles/{article_id}/reject")
async def reject_article(article_id: str, body: RejectBody):
    r = _pb_request(
        "patch",
        f"/api/collections/news_articles/records/{article_id}",
        json={"status": "draft", "review_feedback": body.feedback},
    )
    return r.json()


@app.post("/api/articles/{article_id}/archive")
async def archive_article(article_id: str):
    r = _pb_request(
        "patch",
        f"/api/collections/news_articles/records/{article_id}",
        json={"status": "archived"},
    )
    return r.json()


class PolishBody(BaseModel):
    body: str = ""


@app.post("/api/articles/{article_id}/polish")
async def polish_article(article_id: str, payload: PolishBody | None = None):
    # Use body from request if provided, otherwise fetch from PB
    if payload and payload.body.strip():
        article_body = payload.body
    else:
        article = _pb_request(
            "get", f"/api/collections/news_articles/records/{article_id}"
        ).json()
        article_body = article.get("body", "")
    if not article_body.strip():
        raise HTTPException(status_code=422, detail="Article body is empty")

    res = http_requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": POLISH_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a news editor for Australian croquet. "
                        "Clean up this article body (HTML). "
                        "Fix grammar, punctuation, and awkward phrasing. Trim filler words. "
                        "Keep the meaning, tone, and all facts exactly as submitted. "
                        "Also generate a short punchy headline (8 words or fewer) and a "
                        "1-2 sentence excerpt/summary. "
                        'Return JSON only: {"title": "...", "excerpt": "...", "body": "...cleaned HTML..."}'
                    ),
                },
                {"role": "user", "content": article_body},
            ],
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    data = res.json()
    if "choices" not in data:
        raise HTTPException(status_code=502, detail=f"AI polish failed: {data.get('error', {}).get('message', 'unknown error')}")
    raw = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(raw)
        return {
            "polished": parsed.get("body", raw),
            "title": parsed.get("title"),
            "excerpt": parsed.get("excerpt"),
        }
    except (json.JSONDecodeError, KeyError):
        return {"polished": raw}


@app.get("/api/stats")
async def get_stats():
    counts = {}
    for status in ("submitted", "draft", "published"):
        r = _pb_request(
            "get",
            "/api/collections/news_articles/records",
            params={"filter": f"status='{status}'", "perPage": 1},
        )
        counts[status] = r.json().get("totalItems", 0)
    return counts


@app.post("/api/articles")
async def create_article(body: CreateBody):
    slug = re.sub(r"[^a-z0-9]+", "-", body.title.lower()).strip("-")
    slug = slug[:80]
    r = _pb_request(
        "post",
        "/api/collections/news_articles/records",
        json={
            "title": body.title,
            "body": body.body,
            "excerpt": body.excerpt,
            "category": body.category,
            "status": "draft",
            "slug": slug,
        },
    )
    return r.json()


@app.post("/api/articles/{article_id}/accept")
async def accept_article(article_id: str):
    r = _pb_request(
        "patch",
        f"/api/collections/news_articles/records/{article_id}",
        json={"status": "draft", "review_feedback": ""},
    )
    return r.json()
