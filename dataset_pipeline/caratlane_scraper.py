"""
CaratLane Catalog Scraper for Virtual Try-On Training
------------------------------------------------------
Collects ground-truth paired datasets:
  - Product reference shot (clean white background) -> dataset/caratlane/product/{sku}.jpg
  - Real human model wearing the piece              -> dataset/caratlane/worn/{sku}.jpg
  - SKU & Material Metadata                         -> dataset/caratlane/metadata.jsonl
"""

import os
import re
import json
import time
import random
import logging
import argparse
import requests
from typing import List, Dict, Any, Optional, Tuple

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

logger = logging.getLogger("caratlane_scraper")

CATEGORIES = {
    "necklaces": "https://www.caratlane.com/jewellery/necklaces.html",
    "earrings": "https://www.caratlane.com/jewellery/earrings.html",
    "mangalsutra": "https://www.caratlane.com/jewellery/mangalsutra.html",
}


class CaratLaneScraper:
    def __init__(self, output_dir: str = "dataset/caratlane", delay: float = 0.8):
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
        """Loads already scraped SKUs from metadata.jsonl for seamless resume."""
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
        logger.info(f"Resuming with {len(skus)} existing SKUs already collected.")
        return skus

    def fetch_catalog_page(self, base_url: str, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetches a catalog listing page and parses products from __PRELOADED_STATE__."""
        page_url = f"{base_url}?baseOffset={offset}"
        logger.info(f"Fetching catalog: {page_url}")

        r = self.session.get(page_url, timeout=20)
        if r.status_code != 200:
            logger.warning(f"Failed to fetch {page_url} (status {r.status_code})")
            return []

        idx = r.text.find("__PRELOADED_STATE__")
        if idx == -1:
            logger.warning("Could not locate __PRELOADED_STATE__ in response HTML")
            return []

        brace_start = r.text.find("{", idx)
        decoder = json.JSONDecoder()
        try:
            state, _ = decoder.raw_decode(r.text[brace_start:])
            products = state.get("listingPage", {}).get("listingData", {}).get("products", [])
            return products
        except Exception as e:
            logger.error(f"Failed to decode preloaded state: {e}")
            return []

    def classify_and_extract_images(self, media_list: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts:
        1. Product reference shot (_1_lar.jpg or clean front shot)
        2. Model-worn shot (_3_lar.jpg, _4_lar.jpg, etc. verified by skin percentage)
        """
        urls = [m.get("url", "") for m in media_list if m.get("url", "").endswith((".jpg", ".png", ".webp"))]
        if not urls:
            return None, None

        # 1. Product reference image: CaratLane convention _1_lar.jpg or listfront
        product_url = None
        for u in urls:
            if "_1_lar.jpg" in u:
                product_url = u
                break
        if not product_url:
            for u in urls:
                if "listfront" in u:
                    product_url = u
                    break
        if not product_url and urls:
            product_url = urls[0]

        # 2. Model-worn candidate images: usually _3_lar.jpg, _4_lar.jpg, _5_lar.jpg
        candidate_urls = [u for u in urls if u != product_url and "_lar.jpg" in u]
        worn_url = None
        best_skin_pct = 0.0

        for cand in candidate_urls:
            # Quick probe check: download into memory and calculate skin percentage
            try:
                r_probe = self.session.get(cand, timeout=10)
                if r_probe.status_code == 200:
                    skin_pct = calculate_skin_percentage(r_probe.content)
                    if skin_pct > 20.0 and skin_pct > best_skin_pct:
                        best_skin_pct = skin_pct
                        worn_url = cand
            except Exception:
                continue

        return product_url, worn_url

    def scrape_category(self, category: str = "necklaces", limit: int = 100):
        """Scrapes paired product and model-worn photos for a given category."""
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Available: {list(CATEGORIES.keys())}")

        base_url = CATEGORIES[category]
        offset = 0
        collected_this_run = 0

        logger.info(f"Starting scrape for category: {category} (Target: {limit} pairs)")

        while collected_this_run < limit:
            products = self.fetch_catalog_page(base_url, offset=offset)
            if not products:
                logger.info("No more products returned from catalog.")
                break

            for p in products:
                if collected_this_run >= limit:
                    break

                sku = sanitize_filename(p.get("sku", ""))
                if not sku or sku in self.existing_skus:
                    continue

                name = p.get("name", "Jewellery Item")
                media = p.get("media", [])

                product_url, worn_url = self.classify_and_extract_images(media)

                # We require BOTH clean product shot and real model worn shot to create a paired dataset
                if not product_url or not worn_url:
                    continue

                prod_save_path = os.path.join(self.product_dir, f"{sku}.jpg")
                worn_save_path = os.path.join(self.worn_dir, f"{sku}.jpg")

                p_ok = download_image(product_url, prod_save_path, min_bytes=8000)
                w_ok = download_image(worn_url, worn_save_path, min_bytes=15000)

                if p_ok and w_ok:
                    metadata = {
                        "sku": sku,
                        "name": name,
                        "category": category,
                        "price": p.get("price"),
                        "metal": p.get("metal"),
                        "purity": p.get("purity"),
                        "material": p.get("material"),
                        "product_url": f"https://www.caratlane.com{p.get('url', '')}",
                        "product_image": product_url,
                        "worn_image": worn_url,
                        "product_path": os.path.relpath(prod_save_path, self.output_dir),
                        "worn_path": os.path.relpath(worn_save_path, self.output_dir)
                    }
                    append_jsonl(metadata, self.meta_file)
                    self.existing_skus.add(sku)
                    collected_this_run += 1

                    logger.info(
                        f"[{collected_this_run}/{limit}] Collected pair: {sku} - {name} "
                        f"(Price: Rs. {p.get('price', 'N/A')})"
                    )

                # Respectful delay with jitter
                time.sleep(self.delay + random.uniform(0.1, 0.4))

            offset += len(products)
            time.sleep(self.delay)

        logger.info(
            f"Scrape completed! Collected {collected_this_run} pairs. "
            f"Total dataset size: {len(self.existing_skus)} pairs in {self.output_dir}"
        )


def main():
    parser = argparse.ArgumentParser(description="CaratLane Jewellery Catalog Scraper for Virtual Try-On")
    parser.add_argument("--category", choices=list(CATEGORIES.keys()), default="necklaces", help="Jewellery category")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of paired items to collect")
    parser.add_argument("--output-dir", default="dataset/caratlane", help="Destination directory")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between requests in seconds")
    args = parser.parse_args()

    scraper = CaratLaneScraper(output_dir=args.output_dir, delay=args.delay)
    scraper.scrape_category(category=args.category, limit=args.limit)


if __name__ == "__main__":
    main()
