import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)
print(f"[newsroom] .env path: {_env_path} exists={_env_path.exists()}")
print(f"[newsroom] PB_ADMIN_EMAIL={os.environ.get('PB_ADMIN_EMAIL', 'NOT SET')}")

import logging
import traceback
import requests as http_requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": str(exc)})

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
    if r.status_code == 401:
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


@app.post("/api/articles/{article_id}/polish")
async def polish_article(article_id: str):
    # Fetch current body from PB
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
                        "Return clean HTML only — no commentary, no markdown, no wrapper."
                    ),
                },
                {"role": "user", "content": article_body},
            ],
            "max_tokens": 4096,
        },
        timeout=60,
    )
    data = res.json()
    if "choices" not in data:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {data}")
    return {"polished": data["choices"][0]["message"]["content"]}
