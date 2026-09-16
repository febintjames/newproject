"""
Unified Virtual Try-On Dataset Scraper
--------------------------------------
CLI entry point to scrape high-resolution paired jewellery datasets:
- CaratLane (Titan Group)
- BlueStone

Usage examples:
  python -m dataset_pipeline.unified_scraper --source caratlane --category necklaces --limit 50
  python -m dataset_pipeline.unified_scraper --source bluestone --category necklaces --limit 50
  python -m dataset_pipeline.unified_scraper --source all --limit 100
"""

import sys
import argparse
import logging

try:
    from dataset_pipeline.caratlane_scraper import CaratLaneScraper
    from dataset_pipeline.bluestone_scraper import BlueStoneScraper
except ImportError:
    from caratlane_scraper import CaratLaneScraper
    from bluestone_scraper import BlueStoneScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("unified_scraper")


def main():
    parser = argparse.ArgumentParser(description="Unified High-Jewellery Try-On Scraper")
    parser.add_argument("--source", choices=["caratlane", "bluestone", "all"], default="caratlane",
                        help="Source website to scrape")
    parser.add_argument("--category", choices=["necklaces", "earrings", "all"], default="necklaces",
                        help="Jewellery category")
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum pairs to collect per source/category")
    parser.add_argument("--output-dir", default="dataset",
                        help="Root output directory")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Polite delay between requests in seconds")

    args = parser.parse_args()

    categories = ["necklaces", "earrings"] if args.category == "all" else [args.category]

    if args.source in ["caratlane", "all"]:
        cl_dir = f"{args.output_dir}/caratlane"
        cl_scraper = CaratLaneScraper(output_dir=cl_dir, delay=args.delay)
        for cat in categories:
            logger.info(f"=== Starting CaratLane Scraper for {cat} ===")
            try:
                cl_scraper.scrape_category(category=cat, limit=args.limit)
            except Exception as e:
                logger.error(f"Error in CaratLane {cat}: {e}")

    if args.source in ["bluestone", "all"]:
        bs_dir = f"{args.output_dir}/bluestone"
        bs_scraper = BlueStoneScraper(output_dir=bs_dir, delay=args.delay)
        for cat in categories:
            logger.info(f"=== Starting BlueStone Scraper for {cat} ===")
            try:
                bs_scraper.scrape_category(category=cat, limit=args.limit)
            except Exception as e:
                logger.error(f"Error in BlueStone {cat}: {e}")


if __name__ == "__main__":
    main()
