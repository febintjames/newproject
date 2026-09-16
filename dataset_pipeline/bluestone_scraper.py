"""
BlueStone Catalog Scraper for Virtual Try-On Training
-----------------------------------------------------
Extracts paired high-resolution jewellery images:
  - Clean product shot on neutral background -> dataset/bluestone/product/{sku}.png
  - Human model wearing the piece            -> dataset/bluestone/worn/{sku}.png
  - Metadata                                 -> dataset/bluestone/metadata.jsonl
"""

import os
import re
import json
import time
import random
import logging
import argparse
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple, Set

try:
    from dataset_pipeline.utils import (
        DEFAULT_HEADERS,
        sanitize_filename,
        download_image,
        calculate_skin_percentage,
        append_jsonl
    )
except ImportError:
    from utils import (
        DEFAULT_HEADERS,
        sanitize_filename,
        download_image,
        calculate_skin_percentage,
        append_jsonl
    )

logger = logging.getLogger("bluestone_scraper")

BLUESTONE_CATEGORIES = {
    "necklaces": "https://www.bluestone.com/jewellery/necklaces.html",
    "pendants": "https://www.bluestone.com/jewellery/pendants.html",
    "earrings": "https://www.bluestone.com/jewellery/earrings.html",
}


class BlueStoneScraper:
    def __init__(self, output_dir: str = "dataset/bluestone", delay: float = 1.0):
        self.output_dir = output_dir
        self.delay = delay
        self.product_dir = os.path.join(output_dir, "product")
        self.worn_dir = os.path.join(output_dir, "worn")
        self.meta_file = os.path.join(output_dir, "metadata.jsonl")
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

        os.makedirs(self.product_dir, exist_ok=True)
        os.makedirs(self.worn_dir, exist_ok=True)

        self.existing_skus = self._load_existing_skus()

    def _load_existing_skus(self) -> set:
        skus = set()
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if "sku" in record:
                            skus.add(record["sku"])
                    except Exception:
                        pass
        return skus

    def fetch_listing_page(self, base_url: str, page: int = 1) -> List[str]:
        """Fetches product detail URLs from a catalog page."""
        url = f"{base_url}?page={page}" if page > 1 else base_url
        logger.info(f"Fetching BlueStone catalog: {url}")

        r = self.session.get(url, timeout=20)
        if r.status_code != 200:
            logger.warning(f"Failed to fetch {url} (status {r.status_code})")
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=re.compile(r"/(?:necklaces|pendants|earrings)/.*\.html"))
        clean_links = []
        for a in links:
            href = a.get("href", "")
            if href and not href.startswith("http"):
                href = f"https://www.bluestone.com{href}"
            if href and href not in clean_links:
                clean_links.append(href)

        return clean_links

    def scrape_product_page(self, product_url: str) -> Optional[Dict[str, Any]]:
        """Scrapes a BlueStone product page, finds PICS gallery, and verifies pairs."""
        try:
            r = self.session.get(product_url, timeout=15)
            if r.status_code != 200:
                return None

            soup = BeautifulSoup(r.text, "html.parser")
            title_elem = soup.find("h1") or soup.find("title")
            title = title_elem.get_text(strip=True) if title_elem else "BlueStone Jewellery"

            # Find all high-res PICS images
            raw_imgs = re.findall(r"https://kinclimg\d\.bluestone\.com/[^\"\'\s\\]+PICS-[^\"\'\s\\]+", r.text)
            if not raw_imgs:
                return None

            # Standardize URLs to 1024px high-res
            high_res_imgs: Set[str] = set()
            for u in raw_imgs:
                clean = re.sub(r"\\+$", "", u)
                high_res = re.sub(r"w_\d+", "w_1024", clean)
                high_res_imgs.add(high_res)

            for u in high_res_imgs:
                if "PICS-00000" in u:
                    product_shot = u
                    break

            if not product_shot:
                return None

            # Extract product SKU prefix to filter out carousel images of other products
            m_prefix = re.search(r"giproduct/([A-Z0-9_]+?)_ABCD00-PICS-", product_shot)
            sku_prefix = m_prefix.group(1) if m_prefix else ""

            for u in high_res_imgs:
                if sku_prefix and sku_prefix in u and "PICS-00000" not in u:
                    worn_candidates.append(u)

            if not worn_candidates:
                return None

            # Extract SKU from image pattern: giproduct/{SKU}_...
            m_sku = re.search(r"giproduct/([A-Z0-9]+)_", product_shot)
            sku = m_sku.group(1) if m_sku else sanitize_filename(product_url.split("/")[-1].replace(".html", ""))

            # Find the best model worn image
            worn_shot = None
            best_skin = 0.0

            for cand in worn_candidates:
                try:
                    probe_r = self.session.get(cand, timeout=10)
                    if probe_r.status_code == 200:
                        skin_pct = calculate_skin_percentage(probe_r.content)
                        if skin_pct > 25.0 and skin_pct > best_skin:
                            best_skin = skin_pct
                            worn_shot = cand
                except Exception:
                    continue

            if not worn_shot:
                return None

            return {
                "sku": sku,
                "title": title,
                "product_url": product_url,
                "product_image": product_shot,
                "worn_image": worn_shot,
            }

        except Exception as e:
            logger.debug(f"Error scraping {product_url}: {e}")
            return None

    def scrape_category(self, category: str = "necklaces", limit: int = 50):
        if category not in BLUESTONE_CATEGORIES:
            raise ValueError(f"Unknown category: {category}")

        base_url = BLUESTONE_CATEGORIES[category]
        page = 1
        collected = 0

        logger.info(f"Starting BlueStone scrape for {category} (Target: {limit})")

        while collected < limit:
            prod_links = self.fetch_listing_page(base_url, page=page)
            if not prod_links:
                break

            for url in prod_links:
                if collected >= limit:
                    break

                pair_info = self.scrape_product_page(url)
                if not pair_info:
                    continue

                sku = pair_info["sku"]
                if sku in self.existing_skus:
                    continue

                p_ext = "png" if ".png" in pair_info["product_image"] else "jpg"
                w_ext = "png" if ".png" in pair_info["worn_image"] else "jpg"

                prod_path = os.path.join(self.product_dir, f"{sku}.{p_ext}")
                worn_path = os.path.join(self.worn_dir, f"{sku}.{w_ext}")

                p_ok = download_image(pair_info["product_image"], prod_path, min_bytes=8000)
                w_ok = download_image(pair_info["worn_image"], worn_path, min_bytes=15000)

                if p_ok and w_ok:
                    metadata = {
                        "sku": sku,
                        "name": pair_info["title"],
                        "category": category,
                        "brand": "BlueStone",
                        "product_url": pair_info["product_url"],
                        "product_image": pair_info["product_image"],
                        "worn_image": pair_info["worn_image"],
                        "product_path": os.path.relpath(prod_path, self.output_dir),
                        "worn_path": os.path.relpath(worn_path, self.output_dir)
                    }
                    append_jsonl(metadata, self.meta_file)
                    self.existing_skus.add(sku)
                    collected += 1
                    logger.info(f"[{collected}/{limit}] BlueStone pair saved: {sku} - {pair_info['title'][:40]}")

                time.sleep(self.delay + random.uniform(0.1, 0.4))

            page += 1
            time.sleep(self.delay)

        logger.info(f"BlueStone scrape completed! Total collected: {collected}")


def main():
    parser = argparse.ArgumentParser(description="BlueStone Jewellery Catalog Scraper")
    parser.add_argument("--category", choices=list(BLUESTONE_CATEGORIES.keys()), default="necklaces")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-dir", default="dataset/bluestone")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    scraper = BlueStoneScraper(output_dir=args.output_dir, delay=args.delay)
    scraper.scrape_category(category=args.category, limit=args.limit)


if __name__ == "__main__":
    main()
