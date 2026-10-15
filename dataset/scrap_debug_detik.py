# ini baru ada kompas_url_patterns
import nest_asyncio
nest_asyncio.apply()

import re
import time
import random
import asyncio
import urllib.parse
from datetime import datetime, date, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from playwright.async_api import async_playwright
from tqdm.auto import tqdm

import httpx
import urllib.parse
# ==========================================
# GLOBAL CONFIG
# ==========================================
TARGET = 1000
MAX_DATE = date(2026, 10, 3)
MIN_DATE = date(2025, 10, 3)
OUTPUT_COLUMNS = ["id", "source", "date", "title", "label", "url"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
]

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "mei": "05", "jun": "06",
    "jul": "07", "agu": "08", "sep": "09", "okt": "10", "nov": "11", "des": "12"
}

# ==========================================
# PER-DOMAIN CONFIG
# Each domain shares the same output/filtering structure, while each portal
# has its own scraper because the index-page structures differ.
# ==========================================
DOMAINS = {
    "technology": {
        "label": "Technology",
        "output_file": "data/technology.csv",
        "keywords": ["software", "hardware", "it", "teknologi", "wifi", "internet", "robot", "ict"],
        # CNN: indeks id + the URL section that marks a tech article.
        "cnn_indeks": "teknologi/indeks/8",
        "cnn_section": "/teknologi/",
        # Kompas: ?site= value + the host that tech articles live on.
        "kompas_site": "tekno",
        "kompas_host": "tekno.kompas.com",
		"kompas_url_patterns": [
			"tekno.kompas.com/read/",
			"www.kompas.com/tekno/read/",
		],
        # Detik: desk subdomain + whether that desk uses ISO or MM/DD/YYYY dates.
        "detik_base": "https://inet.detik.com/indeks",
        "detik_host": "inet.detik.com",
        "detik_date_iso": True,   # inet uses YYYY-MM-DD; other desks use MM/DD/YYYY
        # Tribun: one or more index channels (politics has several).
        "tribun_channels": ["index-news/techno"],
        "tribun_marker": "techno",
    },
    "health": {
        "label": "Health",
        "output_file": "data/health.csv",
        "keywords": ["kesehatan", "sehat", "bugar", "obat", "pertolongan", "pencegahan", "vitamin"],
        "cnn_indeks": "gaya-hidup/indeks/10",
        "cnn_section": "/gaya-hidup/",
        "kompas_site": "health",
        "kompas_host": "health.kompas.com",
		"kompas_url_patterns": [
			"health.kompas.com/read/",
			"www.kompas.com/health/read/",
		],
        "detik_base": "https://health.detik.com/indeks",
        "detik_host": "health.detik.com",
        "detik_date_iso": False,
        "tribun_channels": ["index-news/kesehatan"],
        "tribun_marker": "kesehatan",
    },
    "education": {
        "label": "Education",
        "output_file": "data/education.csv",
        "keywords": ["kuliah", "materi", "pendidikan", "kelas", "pembelajaran", "guru", "dosen"],
        "cnn_indeks": "edukasi/indeks/559",
        "cnn_section": "/edukasi/",
        "kompas_site": "edukasi",
        "kompas_host": "www.kompas.com/edu",
		"kompas_url_patterns": [
			"www.kompas.com/edu/read/",
			"edukasi.kompas.com/read/",
		],
        "detik_base": "https://www.detik.com/edu/indeks",
        "detik_host": "detik.com/edu",
        "detik_date_iso": False,
        "tribun_channels": ["pendidikan-go"],
        "tribun_marker": "pendidikan",
    },
    "sport": {
        "label": "Sport",
        "output_file": "data/sport.csv",
        "keywords": ["sepakbola", "bola", "soccer", "nbl", "basket", "volly", "asian games", "skor", "ht ft"],
        "cnn_indeks": "olahraga/indeks/7",
        "cnn_section": "/olahraga/",
        "kompas_site": "bola",
        "kompas_host": "bola.kompas.com",
		"kompas_url_patterns": [
			"bola.kompas.com/read/",
			"www.kompas.com/sports/read/",
		],
        "detik_base": "https://sport.detik.com/indeks",
        "detik_host": "sport.detik.com",
        "detik_date_iso": False,
        "tribun_channels": ["index-news/sport"],
        "tribun_marker": "sport",
    },
    "politics": {
        "label": "Politics",
        "output_file": "data/politic.csv",
        "keywords": ["politik", "parpol", "partai", "pemilu", "kpu", "pencoblosan", "pemungutan suara", "oposisi"],
        "cnn_indeks": "nasional/indeks/3",
        "cnn_section": "/nasional/",
        "kompas_site": "nasional",
        "kompas_host": "nasional.kompas.com",
		"kompas_url_patterns": [
			"nasional.kompas.com/read/",
			"www.kompas.com/nasional/read/",
			"sorotpolitik.kompas.com/read/",
		],
        "detik_base": "https://news.detik.com/indeks",
        "detik_host": "news.detik.com",
        "detik_date_iso": False,
        "tribun_channels": [
            "index-news/nasional",
            "index-news/dpd-ri",
            "index-news/mpr-ri",
            "index-news/kilas-kementerian",
        ],
        "tribun_marker": "nasional",
    },
}


# ==========================================
# SHARED HELPERS
# ==========================================
def keyword_matches(keyword, text):
    """True only if `keyword` appears as a WHOLE WORD in `text` (case-insensitive).
    Multi-word keywords may be separated by space or hyphen."""
    words = [re.escape(w) for w in keyword.lower().split()]
    pattern = r"\b" + r"[\s\-]+".join(words) + r"\b"
    return re.search(pattern, text.lower()) is not None


def title_has_keyword(title, keywords):
    """Secondary domain guard: the ARTICLE TITLE must contain a domain keyword."""
    return any(keyword_matches(k, title) for k in keywords)


def new_funnel():
    """Per-portal drop-off counters."""
    return {
        "candidates": 0,
        "date_rejected": 0,
        "keyword_rejected": 0,
        "duplicate": 0,
        "collected": 0,
    }


def print_funnel(portal_name, f):
    """One funnel summary line (kept as a tiny helper only because it is reused
    identically by all four scrapers)."""
    print(
        f"[FUNNEL] {portal_name} | candidates={f['candidates']} | "
        f"collected={f['collected']} | "
        f"reject(date={f['date_rejected']}, "
        f"keyword={f['keyword_rejected']}, "
        f"duplicate={f['duplicate']})",
        flush=True,
    )


def extract_detik_date(date_elem):
    """Parse a Detik listing date element into YYYY-MM-DD (same method as edu)."""
    fallback = datetime.now().strftime("%Y-%m-%d")
    if not date_elem:
        return fallback
    raw = re.sub(r"\s+", " ", date_elem.get("title") or date_elem.text).strip()
    if "yang lalu" in raw:
        return fallback
    try:
        match = re.search(r"(\d{1,2})\s+([a-zA-Z]{3})\s+(\d{4})", raw)
        if match:
            day = f"{int(match.group(1)):02d}"
            mon = MONTH_MAP.get(match.group(2).lower()[:3], "01")
            year = match.group(3)
            return f"{year}-{mon}-{day}"
    except Exception:
        pass
    return fallback


def clean_title(title):
    if not title:
        return None
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"\s+\d{6,}$", "", title)
    title = re.sub(r"^(FOTO:|VIDEO:|INFOGRAFIS:|BREAKING NEWS:|Ilustrasi|Gambar|Kumpulan)\s*[:\-]?\s*", "", title, flags=re.I)
    return title if len(title) > 15 else None


def within_window(dstr):
    try:
        y, m, d = map(int, dstr.split("-"))
        return MIN_DATE <= date(y, m, d) <= MAX_DATE
    except Exception:
        return False


def format_excel(file_path):
    """Workbook styling (same as the edu notebook)."""
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    ws.views.sheetView[0].showGridLines = True
    ws.freeze_panes = "A2"
    font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    font_body = Font(name="Segoe UI", size=10)
    fill_header = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    grid_border = Border(
        left=Side(style="thin", color="D3D3D3"),
        right=Side(style="thin", color="D3D3D3"),
        top=Side(style="thin", color="D3D3D3"),
        bottom=Side(style="thin", color="D3D3D3")
    )
    for cell in ws[1]:
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.font = font_body
            cell.border = grid_border
            cell.alignment = Alignment(horizontal="center" if cell.column in [1, 3] else "left", vertical="center")
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = 65 if col_letter in ["D", "E"] else max(max_len + 3, 14)
    wb.save(file_path)


# ==========================================
# DETIK (index page — correct URL format per desk)
# ==========================================
# ==========================================
# DETIK (Playwright — index is client-side rendered)
# ==========================================
async def scrape_detik(cfg, articles, seen_urls, seen_titles, target, bar=None, funnel=None):
    """
    Detik scraper using the per-desk INDEX page.

    Detik's index is a client-side rendered SPA. URL-based ?page=N navigation
    is unreliable — the server returns page-1 HTML and the JS may or may not
    swap the list. The only robust method is to load page 1, then CLICK the
    pagination link, letting the SPA's own router handle the transition.
    """
    portal_name = "Detik"
    f = funnel if funnel is not None else new_funnel()
    keywords = cfg["keywords"]
    host = cfg["detik_host"]
    use_iso = cfg["detik_date_iso"]
    base = f"https://{host}/indeks"
    print(f"\n[DETIK] Starting with Playwright (click pagination)...", flush=True)

    current = MAX_DATE
    fails = 0
    MAX_PAGES_PER_DATE = 30

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            user_agent=USER_AGENTS[0],
            locale="id-ID",
            timezone_id="Asia/Jakarta",
            viewport={"width": 1366, "height": 768},
            extra_http_headers={
                "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Referer": "https://www.detik.com/",
                "Sec-Fetch-Site": "same-site",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
                "Upgrade-Insecure-Requests": "1",
                "DNT": "1",
            },
        )
        page = await context.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['id-ID', 'id', 'en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {name: 'Chrome PDF Plugin'},
                    {name: 'Chrome PDF Viewer'},
                    {name: 'Native Client'}
                ]
            });
            window.chrome = {runtime: {}, loadTimes: function(){}, csi: function(){}};
            Object.defineProperty(navigator, 'permissions', {
                get: () => ({query: () => Promise.resolve({state: 'granted'})})
            });
        """)

        while current >= MIN_DATE and len(articles) < target and fails < 5:
            date_iso = current.strftime("%Y-%m-%d")
            if use_iso:
                url_date = current.strftime("%Y-%m-%d")
            else:
                url_date = current.strftime("%m/%d/%Y")

            # Load page 1 ONCE per date. We'll click from here.
            url = f"{base}?page=1&date={url_date}"
            print(f"[INDEX] {portal_name} | Date={date_iso} | Page=1", flush=True)

            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await asyncio.sleep(random.uniform(2, 3))

                # Wait for the JS-rendered list to appear and stabilise.
                prev_first = None
                for _ in range(8):
                    try:
                        first_a = await page.query_selector('a[href*="/d-"]')
                        cur_first = await first_a.get_attribute("href") if first_a else None
                    except Exception:
                        cur_first = None
                    if cur_first and cur_first == prev_first:
                        break
                    prev_first = cur_first
                    await asyncio.sleep(1.2)

                content = await page.content()
                if "gokuProps" in content or len(content) < 2000:
                    fails += 1
                    print(f"[DETIK] blocked {date_iso} p1 ({fails}/5)", flush=True)
                    current -= timedelta(days=1)
                    continue

                fails = 0
                page_num = 1
                date_exhausted = False

                while len(articles) < target and page_num <= MAX_PAGES_PER_DATE:
                    # ---------- scrape current DOM ----------
                    new = 0
                    raw_links = 0
                    seen_in_page = set()

                    # Pass 1: collect every unique article URL, keeping the
                    # anchor with the longest text (thumbnail anchors are empty;
                    # headline anchors carry the real title).
                    best = {}
                    links = await page.query_selector_all('a[href*="/d-"]')
                    for link in links:
                        href = await link.get_attribute("href")
                        if not href:
                            continue
                        art_url = urllib.parse.urljoin(f"https://{host}/", href).split("?")[0].strip()
                        if host not in art_url:
                            continue
                        txt = (await link.inner_text()).strip()
                        if art_url not in best or len(txt) > len(best[art_url]):
                            best[art_url] = txt

                    # Pass 2: apply the funnel on the deduped URL→title map.
                    for art_url, raw_title in best.items():
                        if len(articles) >= target:
                            break
                        raw_links += 1
                        f["candidates"] += 1

                        if art_url in seen_urls:
                            f["duplicate"] += 1
                            continue

                        title_text = clean_title(re.sub(r"\s+", " ", raw_title))
                        if not title_text:
                            continue
                        if title_text.lower() in seen_titles:
                            f["duplicate"] += 1
                            continue

                        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})", art_url)
                        dstr = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else date_iso
                        if not within_window(dstr):
                            f["date_rejected"] += 1
                            continue
                        if not title_has_keyword(title_text, keywords):
                            f["keyword_rejected"] += 1
                            continue

                        seen_urls.add(art_url)
                        seen_titles.add(title_text.lower())
                        articles.append({
                            "id": len(articles) + 1,
                            "source": "Detik",
                            "date": dstr,
                            "title": title_text,
                            "label": None,
                            "url": art_url,
                        })
                        f["collected"] += 1
                        new += 1
                        if bar is not None:
                            bar.update(1)

                    print(
                        f"[INDEX] {portal_name} | Date={date_iso} Page={page_num} | "
                        f"raw_links={raw_links} | new={new} | "
                        f"Collected={len(articles)}/{target} | Remaining={target - len(articles)}",
                        flush=True,
                    )

                    # ---------- navigate to next page via URL ----------
                    # Detik does a full page reload on ?page=N. Compare the
                    # ENTIRE set of article URLs before/after to decide whether
                    # the page actually changed — comparing only the first URL
                    # fails when a pinned/featured article stays visible on
                    # every page.
                    urls_before = set(best.keys())

                    next_url = f"{base}?page={page_num + 1}&date={url_date}"
                    print(f"[DETIK] navigating to {next_url}", flush=True)

                    await page.goto(next_url, timeout=30000, wait_until="domcontentloaded")
                    await asyncio.sleep(random.uniform(2, 3))

                    # Poll until the URL set stabilises AND is different from before.
                    dom_changed = False
                    prev_snapshot = None
                    for _ in range(20):                       # 20 × 1.5s = 30s
                        await asyncio.sleep(1.5)
                        try:
                            anchors = await page.query_selector_all('a[href*="/d-"]')
                            urls_now = set()
                            for a in anchors:
                                h = await a.get_attribute("href")
                                if h:
                                    u = urllib.parse.urljoin(f"https://{host}/", h).split("?")[0].strip()
                                    if host in u:
                                        urls_now.add(u)
                        except Exception:
                            urls_now = set()

                        # wait for list to stabilise (same set twice in a row)
                        if urls_now and urls_now == prev_snapshot:
                            # stable — now check if it differs from page 1
                            if urls_now != urls_before:
                                dom_changed = True
                            break
                        prev_snapshot = urls_now

                    if not dom_changed:
                        print(
                            f"[DETIK] {date_iso} p{page_num + 1} — same URL set as "
                            f"p{page_num}, date exhausted",
                            flush=True,
                        )
                        date_exhausted = True
                        break

                    page_num += 1

                if date_exhausted:
                    current -= timedelta(days=1)

            except Exception as e:
                print(f"[DETIK] error {date_iso}: {e}", flush=True)
                fails += 1
                current -= timedelta(days=1)

        await browser.close()
    print_funnel(portal_name, f)


# ==========================================
# CNN (date + page)
# ==========================================
def scrape_cnn(cfg, articles, seen_urls, seen_titles, target, bar=None, funnel=None):
    portal_name = "CNN Indonesia"
    f = funnel if funnel is not None else new_funnel()
    keywords = cfg["keywords"]
    section = cfg["cnn_section"]
    base = f"https://www.cnnindonesia.com/{cfg['cnn_indeks']}"
    print(f"\n[CNN] Starting (date+page method)...", flush=True)
    current = MAX_DATE
    while current >= MIN_DATE and len(articles) < target:
        date_slash = current.strftime("%Y/%m/%d")
        date_iso = current.strftime("%Y-%m-%d")
        page = 1
        empty_streak = 0
        while len(articles) < target and empty_streak < 2:
            url = f"{base}?date={date_slash}&page={page}"
            print(f"[INDEX] {portal_name} | Date={date_iso} | Page={page}", flush=True)
            headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "id,en;q=0.9"}
            try:
                time.sleep(random.uniform(1.0, 2.0))
                res = requests.get(url, headers=headers, timeout=12)
                if res.status_code != 200 or "gokuProps" in (res.text or "")[:1500]:
                    print(f"[INDEX] {portal_name} | Date={date_iso} Page={page} | blocked/empty (HTTP {res.status_code})", flush=True)
                    empty_streak += 1
                    page += 1
                    continue
                soup = BeautifulSoup(res.content, "html.parser")

                # Scope to <article> tags so sidebar / trending / footer links
                # don't inflate raw_links. CNN's listing items are wrapped in
                # <article>; a page with 0 <article> tags genuinely has 0 articles.
                article_nodes = soup.select("article")
                if not article_nodes:
                    print(
                        f"[INDEX] {portal_name} | Date={date_iso} Page={page} | "
                        f"raw_links=0 | new=0 | "
                        f"Collected={len(articles)}/{target} | Remaining={target - len(articles)}",
                        flush=True,
                    )
                    empty_streak += 1
                    page += 1
                    continue

                # Dedupe hrefs within the page — each article has multiple <a> tags
                # (headline, thumbnail, tag) that all point to the same URL.
                seen_in_page = set()
                candidate_anchors = []
                for node in article_nodes:
                    for a in node.find_all("a", href=True):
                        h = a["href"].split("?")[0]
                        if h in seen_in_page:
                            continue
                        seen_in_page.add(h)
                        candidate_anchors.append(a)

                new = 0
                raw_links = 0
                for a in candidate_anchors:
                    if len(articles) >= target:
                        break
                    href = a["href"].split("?")[0]
                    if section not in href:
                        continue
                    raw_links += 1
                    f["candidates"] += 1
                    title_text = clean_title(re.sub(r"\s+", " ", a.get_text()).strip())
                    if not title_text:
                        continue
                    if href in seen_urls or title_text.lower() in seen_titles:
                        f["duplicate"] += 1
                        continue
                    # date from URL preferred
                    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})", href)
                    dstr = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else date_iso
                    if not within_window(dstr):
                        f["date_rejected"] += 1
                        continue
                    # Secondary guard: keyword must appear in the TITLE.
                    if not title_has_keyword(title_text, keywords):
                        f["keyword_rejected"] += 1
                        continue
                    seen_urls.add(href)
                    seen_titles.add(title_text.lower())
                    articles.append({
                        "id": len(articles) + 1,
                        "source": "CNN Indonesia",
                        "date": dstr,
                        "title": title_text,
                        "label": None,
                        "url": href
                    })
                    f["collected"] += 1
                    new += 1
                    if bar is not None:
                        bar.update(1)
                print(
                    f"[INDEX] {portal_name} | Date={date_iso} Page={page} | "
                    f"raw_links={raw_links} | new={new} | "
                    f"Collected={len(articles)}/{target} | Remaining={target - len(articles)}",
                    flush=True,
                )
                # Keep paging while the page still HAD article links; a page with
                # 0 links is the real end of this date. (new==0 alone can just
                # mean duplicates / keyword-filtered.)
                if raw_links == 0:
                    empty_streak += 1
                else:
                    empty_streak = 0
                page += 1
            except Exception as e:
                print(f"[CNN] error: {e}", flush=True)
                break
        current -= timedelta(days=1)
    print_funnel(portal_name, f)


# ==========================================
# KOMPAS (Playwright, date + page)
# ==========================================
async def scrape_kompas_playwright(cfg, articles, seen_urls, seen_titles, target, bar=None, funnel=None):
    portal_name = "Kompas"
    f = funnel if funnel is not None else new_funnel()
    keywords = cfg["keywords"]
    site = cfg["kompas_site"]
    host = cfg["kompas_host"]
    print(f"\n[KOMPAS] Starting with Playwright (async)...", flush=True)
    current = MAX_DATE
    fails = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="id-ID",
            viewport={"width": 1366, "height": 768},
        )
        page = await context.new_page()

        while current >= MIN_DATE and len(articles) < target and fails < 5:
            date_iso = current.strftime("%Y-%m-%d")
            page_num = 1
            # Paginate within the date: keep going while the page still has
            # article links. Stop the date only when a page has NO links at all.
            while len(articles) < target and fails < 5:
                url = f"https://indeks.kompas.com/?site={site}&date={date_iso}&page={page_num}"
                print(f"[INDEX] {portal_name} | Date={date_iso} | Page={page_num}", flush=True)

                try:
                    await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    await asyncio.sleep(random.uniform(2, 3))

                    # Scroll to trigger lazy-loading of the article list. Stops
                    # early once the /read/ link count stops growing.
                    prev = 0
                    for _ in range(6):
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await asyncio.sleep(1.2)
                        cur = await page.evaluate(
                            'document.querySelectorAll(\'a[href*="/read/"]\').length'
                        )
                        if cur == prev:
                            break
                        prev = cur
                    await page.evaluate("window.scrollTo(0, 0)")
                    await asyncio.sleep(0.4)

                    content = await page.content()
                    if "gokuProps" in content or len(content) < 2000:
                        fails += 1
                        print(f"[KOMPAS] blocked {date_iso} p{page_num} ({fails}/5)", flush=True)
                        break  # move to previous date

                    fails = 0
                    new = 0

                    patterns = cfg.get("kompas_url_patterns") or [f"{cfg['kompas_host']}/read/"]
                    css_sel = ", ".join(f'a[href*="{p}"]' for p in patterns)

                    links = await page.query_selector_all(css_sel)

                    new = 0
                    raw_links = 0
                    seen_in_page = set()  # one article often has 2+ anchors on the same page
                    for link in links:
                        if len(articles) >= target:
                            break
                        href = await link.get_attribute("href")
                        if not href:
                            continue

                        href = urllib.parse.urljoin("https://indeks.kompas.com/", href).split("?")[0]

                        # (1) in-page dedupe — one article often has 2+ anchors
                        #     on the same page. This is what fixed politics (22 → 11).
                        if href in seen_in_page:
                            continue
                        seen_in_page.add(href)

                        # (2) domain guard (rejects money.kompas.com etc.)
                        if not any(p in href for p in patterns):
                            continue

                        # (3) raw_links = article cards on THIS page → count HERE,
                        #     BEFORE the cross-page dedupe. That is the point of "raw".
                        raw_links += 1
                        f["candidates"] += 1

                        # (4) cross-page dedupe — page 2+ sidebar links back to
                        #     page 1 articles; those shouldn't inflate `new`, but
                        #     they DO count as raw links above.
                        if href in seen_urls:
                            f["duplicate"] += 1
                            continue

                        title_text = clean_title((await link.inner_text()).strip())
                        if not title_text:
                            continue
                        if title_text.lower() in seen_titles:
                            f["duplicate"] += 1
                            continue

                        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})", href)
                        if m:
                            dstr = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                            if not within_window(dstr):
                                f["date_rejected"] += 1
                                continue
                        else:
                            dstr = date_iso

                        if not title_has_keyword(title_text, keywords):
                            f["keyword_rejected"] += 1
                            continue

                        seen_urls.add(href)
                        seen_titles.add(title_text.lower())
                        articles.append({
                            "id": len(articles) + 1,
                            "source": "Kompas",
                            "date": dstr,
                            "title": title_text,
                            "label": None,
                            "url": href,
                        })
                        f["collected"] += 1
                        new += 1
                        if bar is not None:
                            bar.update(1)

                    print(
                        f"[INDEX] {portal_name} | Date={date_iso} Page={page_num} | "
                        f"raw_links={raw_links} | new={new} | "
                        f"Collected={len(articles)}/{target} | Remaining={target - len(articles)}",
                        flush=True,
                    )

                    # Advance pages while the page still HAD article links.
                    if raw_links == 0:
                        break  # genuinely no articles on this page -> next date
                    page_num += 1

                except Exception as e:
                    print(f"[KOMPAS] error: {e}", flush=True)
                    fails += 1
                    break

            current -= timedelta(days=1)

        await browser.close()
    print_funnel(portal_name, f)


# ==========================================
# TRIBUN (Playwright, page-based; one or more channels)
# ==========================================
async def scrape_tribun_playwright(cfg, articles, seen_urls, seen_titles, target, bar=None, funnel=None):
    portal_name = "Tribun News"
    f = funnel if funnel is not None else new_funnel()
    keywords = cfg["keywords"]
    marker = cfg["tribun_marker"]
    print(f"\n[TRIBUN] Starting with Playwright (async)...", flush=True)
    fails = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="id-ID",
            viewport={"width": 1366, "height": 768},
        )
        page = await context.new_page()

        # Politics has several channels; others have one. Same logic per channel.
        for channel in cfg["tribun_channels"]:
            page_num = 1
            while len(articles) < target and fails < 5:
                url = f"https://www.tribunnews.com/{channel}?page={page_num}"
                print(f"[INDEX] {portal_name} | Channel={channel} | Page={page_num}", flush=True)

                try:
                    await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    await asyncio.sleep(random.uniform(2, 3))

                    content = await page.content()
                    if "gokuProps" in content or len(content) < 2000:
                        fails += 1
                        print(f"[TRIBUN] blocked {channel} page {page_num} ({fails}/5)", flush=True)
                        page_num += 1
                        continue

                    fails = 0
                    new = 0
                    raw_links = 0

                    ART_ID_RE = re.compile(r"^/[\w\-]+/\d+/")   # e.g. /techno/7891945/slug

                    links = await page.query_selector_all('a[href*="tribunnews.com"]')
                    seen_in_page = set()  # headline + thumbnail → same URL twice on this page
                    for link in links:
                        if len(articles) >= target:
                            break
                        href = await link.get_attribute("href")
                        if not href:
                            continue
                        href = href.split("?")[0].split("#")[0]

                        # Count each URL once per page, regardless of later filters.
                        if href in seen_in_page:
                            continue
                        seen_in_page.add(href)

                        # 1. Must belong to this channel
                        if marker not in href:
                            continue

                        # 2. Must look like a real article: /channel/NUMERIC_ID/slug
                        path = urllib.parse.urlparse(href).path
                        if not ART_ID_RE.match(path):
                            continue

                        # Count article cards on this page BEFORE cross-page dedupe
                        raw_links += 1
                        f["candidates"] += 1

                        # Cross-page dedupe (page 2+ sidebar links back to page 1)
                        if href in seen_urls:
                            f["duplicate"] += 1
                            continue

                        title_text = clean_title((await link.inner_text()).strip())
                        if not title_text:
                            continue
                        if title_text.lower() in seen_titles:
                            f["duplicate"] += 1
                            continue

                        # 4. No date in URL → use today's date as a fallback, but still bound it.
                        #    (If you want real publish dates you'll need to fetch the article page —
                        #     see the note below.)
                        dstr = datetime.now().strftime("%Y-%m-%d")
                        if not within_window(dstr):
                            f["date_rejected"] += 1
                            continue

                        if not title_has_keyword(title_text, keywords):
                            f["keyword_rejected"] += 1
                            continue

                        seen_urls.add(href)
                        seen_titles.add(title_text.lower())
                        articles.append({
                            "id": len(articles) + 1,
                            "source": "Tribun News",
                            "date": dstr,
                            "title": title_text,
                            "label": None,
                            "url": href,
                        })
                        f["collected"] += 1
                        new += 1
                        if bar is not None:
                            bar.update(1)

                    print(
                        f"[INDEX] {portal_name} | Channel={channel} Page={page_num} | "
                        f"raw_links={raw_links} | new={new} | "
                        f"Collected={len(articles)}/{target} | Remaining={target - len(articles)}",
                        flush=True,
                    )
                    # Stop this channel only when a page has NO matching links.
                    if raw_links == 0:
                        break
                    page_num += 1

                except Exception as e:
                    print(f"[TRIBUN] error: {e}", flush=True)
                    fails += 1
                    page_num += 1

        await browser.close()
    print_funnel(portal_name, f)


# ==========================================
# RUN ONE DOMAIN
# ==========================================
async def run_domain(domain_key):
    cfg = DOMAINS[domain_key]
    articles = []
    seen_urls = set()
    seen_titles = set()
    funnels = {}

    print(f"\n========== Starting {cfg['label']} scraper (Target: {TARGET}) ==========", flush=True)
    bar = tqdm(total=TARGET, desc=f"Harvesting {cfg['label']}", unit="rows")

    def fn(name):
        funnels[name] = new_funnel()
        return funnels[name]

    await scrape_detik(cfg, articles, seen_urls, seen_titles, TARGET, bar, fn("Detik"))
    if len(articles) < TARGET:
        await scrape_kompas_playwright(cfg, articles, seen_urls, seen_titles, TARGET, bar, fn("Kompas"))
    if len(articles) < TARGET:
        scrape_cnn(cfg, articles, seen_urls, seen_titles, TARGET, bar, fn("CNN Indonesia"))  # unchanged
    if len(articles) < TARGET:
        await scrape_tribun_playwright(cfg, articles, seen_urls, seen_titles, TARGET, bar, fn("Tribun News"))

    bar.close()

    print(f"\n[FUNNEL] {cfg['label']} drop-off by portal:", flush=True)
    for name, f in funnels.items():
        print_funnel(name, f)

    # finalize: exactly TARGET rows, contiguous ids from 1, empty label.
    articles = articles[:TARGET]
    for i, row in enumerate(articles, 1):
        row["id"] = i

    import os
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(articles, columns=OUTPUT_COLUMNS)
    df["label"] = pd.NA
    df.to_csv(cfg["output_file"], index=False)
    print(f"[SUCCESS] Saved {len(df)} rows → {cfg['output_file']}", flush=True)

    xlsx = cfg["output_file"].replace(".csv", ".xlsx")
    df.to_excel(xlsx, index=False)
    try:
        format_excel(xlsx)
        print(f"[SUCCESS] Excel also saved → {xlsx}", flush=True)
    except Exception:
        pass
    return df


# ==========================================
# RUN ALL 5 DOMAINSA
# ==========================================
async def run_all():
    for domain_key in DOMAINS:
        await run_domain(domain_key)


# In a Jupyter/Colab cell you can also run a single domain:
#     await run_domain("technology")
# or all five:
# await run_all()
await run_domain("technology")
# await run_domain("health")
# await run_domain("education")
# await run_domain("politics")
# await run_domain("sport")