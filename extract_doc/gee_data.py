import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.async_api import async_playwright
import os

BASE_URL = "https://developers.google.com/earth-engine/apidocs"
DOCS_DIR = "gee_api_docs_txt"
os.makedirs(DOCS_DIR, exist_ok=True)

async def fetch_all_links(page):
    await page.goto(BASE_URL)
    await page.wait_for_selector('div.devsite-book-nav', timeout=60000)
    await page.screenshot(path="debug_sidebar.png", full_page=True)  # Optional debug

    content = await page.content()
    soup = BeautifulSoup(content, "html.parser")

    sidebar = soup.find("div", class_="devsite-book-nav")
    if not sidebar:
        print("❌ Sidebar not found.")
        return []

    links = []
    for a_tag in sidebar.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(BASE_URL, href)
        if full_url.startswith(BASE_URL):
            links.append(full_url)

    return list(set(links))

async def fetch_and_save(page, url, index):
    try:
        await page.goto(url)
        await page.wait_for_selector('div.devsite-article-body', timeout=60000)
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

        content_div = soup.find("div", class_="devsite-article-body clearfix")
        if not content_div:
            print(f"⚠️ No main content at {url}")
            return

        text_parts = []

        # Title
        title_tag = soup.find("h1")
        if title_tag:
            text_parts.append(f"# {title_tag.get_text(strip=True)}")

        # Main text content
        text_parts.append(content_div.get_text(separator="\n", strip=True))

        # devsite-selector blocks (Python + JS code)
        for selector in content_div.find_all("devsite-selector"):
            for pre in selector.find_all("pre"):
                lang = pre.get("data-language-code", "unknown").strip()
                code = pre.get_text(strip=True)
                text_parts.append(f"\n--- {lang.upper()} CODE BLOCK ---\n{code}")

        # Extra <pre> code blocks not inside devsite-selector
        for pre in content_div.find_all("pre"):
            if not pre.find_parent("devsite-selector"):
                pre_text = pre.get_text(strip=True)
                text_parts.append("\n--- CODE BLOCK ---\n" + pre_text)

        combined_text = "\n\n".join(text_parts)
        filename = os.path.join(DOCS_DIR, f"api_{index:03}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"✅ Saved: {filename}")

    except Exception as e:
        print(f"❌ Error at {url}: {e}")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        links = await fetch_all_links(page)
        print(f"🔗 Found {len(links)} API doc links.")
        await browser.close()

        # Fetch rendered content for each URL using a Playwright page
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        for i, url in enumerate(links):
            await fetch_and_save(page, url, i)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
