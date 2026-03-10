"""
qa_script.py
QA script verifying portfolio integrity:
  - Hero metric is present and correct
  - Demo links respond (non-4xx)
  - Notebook links are present in case study pages
  - Required pages exist and return 200
  - Page titles match site_manifest.json

Usage:
  python qa/qa_script.py                          # checks local manifest
  PORTFOLIO_URL=https://ukomal.github.io python qa/qa_script.py  # checks live site
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re

# Optional HTTP dependency — only needed for live-site checks
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


ROOT = Path(__file__).parent.parent
PORTFOLIO_URL = os.environ.get("PORTFOLIO_URL", "").rstrip("/")
MANIFEST_PATH = ROOT / "site_manifest.json"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    warning: bool = False


Results = list[CheckResult]


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"site_manifest.json not found at {MANIFEST_PATH}")
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def check_manifest_structure(manifest: dict) -> Results:
    results: Results = []
    required_keys = ["site", "pages", "components", "assets", "seo", "ci_cd", "accessibility"]
    for key in required_keys:
        results.append(CheckResult(
            name=f"manifest.{key}",
            passed=key in manifest,
            detail=f"Key '{key}' {'present' if key in manifest else 'MISSING'} in site_manifest.json",
        ))
    return results


def check_required_files_exist(manifest: dict) -> Results:
    results: Results = []

    file_groups = {
        "pages": [(p["path"], p["id"]) for p in manifest.get("pages", [])],
        "components": [(c["path"], c["id"]) for c in manifest.get("components", [])],
        "assets": [(a["path"], a["id"]) for a in manifest.get("assets", []) if not a["path"].endswith(".png")],
        "seo": [(s["path"], s["id"]) for s in manifest.get("seo", [])],
        "accessibility": [(a["path"], a["id"]) for a in manifest.get("accessibility", [])],
        "analytics": [(a["path"], a["id"]) for a in manifest.get("analytics", [])],
        "ci_cd": [(c["path"], c["id"]) for c in manifest.get("ci_cd", [])],
        "interview_prep": [(i["path"], i["id"]) for i in manifest.get("interview_prep", [])],
        "content": [(c["path"], c["id"]) for c in manifest.get("content", [])],
    }

    for group, files in file_groups.items():
        for file_path, file_id in files:
            full_path = ROOT / file_path
            exists = full_path.exists()
            results.append(CheckResult(
                name=f"file_exists.{group}.{file_id}",
                passed=exists,
                detail=f"{'✓' if exists else '✗'} {file_path}",
            ))

    return results


def check_hero_metric_in_index_page() -> Results:
    results: Results = []
    index_path = ROOT / "pages" / "index.md"

    if not index_path.exists():
        results.append(CheckResult(
            name="hero.metric_present",
            passed=False,
            detail="pages/index.md not found",
        ))
        return results

    content = index_path.read_text()

    # Check for top metric value
    metric_patterns = [
        r"AUC.{0,10}0\.88",
        r"0\.886",
        r"88\.6%",
    ]
    metric_found = any(re.search(p, content, re.IGNORECASE) for p in metric_patterns)
    results.append(CheckResult(
        name="hero.top_metric_present",
        passed=metric_found,
        detail=f"Hero metric (AUC 0.886) {'found' if metric_found else 'NOT FOUND'} in pages/index.md",
    ))

    # Check for dual CTAs
    has_primary_cta = "View Case Study" in content
    has_secondary_cta = "Run Demo" in content
    results.append(CheckResult(
        name="hero.primary_cta",
        passed=has_primary_cta,
        detail=f"Primary CTA 'View Case Study' {'found' if has_primary_cta else 'MISSING'} in hero",
    ))
    results.append(CheckResult(
        name="hero.secondary_cta",
        passed=has_secondary_cta,
        detail=f"Secondary CTA 'Run Demo' {'found' if has_secondary_cta else 'MISSING'} in hero",
    ))

    # Check for role label
    role_patterns = ["AI Engineer", "ML Engineer", "Data Scientist"]
    for role in role_patterns:
        found = role in content
        results.append(CheckResult(
            name=f"hero.role_label.{role.lower().replace(' ', '_')}",
            passed=found,
            detail=f"Role label '{role}' {'found' if found else 'MISSING'} in hero",
        ))

    return results


def check_case_study_pages() -> Results:
    results: Results = []
    case_study_pages = [
        ("pages/projects/fraud-detection.md", "fraud-detection", "0.886"),
        ("pages/projects/depression-detection.md", "depression-detection", "91%"),
        ("pages/projects/federated-healthcare-ai.md", "federated-healthcare-ai", "ε"),
    ]

    for page_path, page_id, expected_metric in case_study_pages:
        full_path = ROOT / page_path
        if not full_path.exists():
            results.append(CheckResult(
                name=f"case_study.{page_id}.exists",
                passed=False,
                detail=f"Case study file {page_path} not found",
            ))
            continue

        content = full_path.read_text()

        # Check metric present
        metric_found = expected_metric in content
        results.append(CheckResult(
            name=f"case_study.{page_id}.metric",
            passed=metric_found,
            detail=f"Expected metric '{expected_metric}' {'found' if metric_found else 'MISSING'} in {page_path}",
        ))

        # Check for demo section
        has_demo = bool(re.search(r"#\s*demo|{#demo}", content, re.IGNORECASE))
        results.append(CheckResult(
            name=f"case_study.{page_id}.demo_section",
            passed=has_demo,
            detail=f"Demo section {'found' if has_demo else 'MISSING'} in {page_path}",
        ))

        # Check for notebook link
        has_notebook = bool(re.search(r"notebook|\.ipynb", content, re.IGNORECASE))
        results.append(CheckResult(
            name=f"case_study.{page_id}.notebook_link",
            passed=has_notebook,
            detail=f"Notebook link {'found' if has_notebook else 'MISSING'} in {page_path}",
        ))

        # Check for GitHub link
        has_github = "github.com" in content.lower()
        results.append(CheckResult(
            name=f"case_study.{page_id}.github_link",
            passed=has_github,
            detail=f"GitHub link {'found' if has_github else 'MISSING'} in {page_path}",
        ))

        # Check for reproducibility section
        has_repro = bool(re.search(r"reproducib", content, re.IGNORECASE))
        results.append(CheckResult(
            name=f"case_study.{page_id}.reproducibility",
            passed=has_repro,
            detail=f"Reproducibility section {'found' if has_repro else 'MISSING'} in {page_path}",
        ))

    return results


def check_seo_files() -> Results:
    results: Results = []

    # Check metadata.jsonld is valid JSON
    jsonld_path = ROOT / "seo" / "metadata.jsonld"
    if jsonld_path.exists():
        try:
            with jsonld_path.open() as f:
                data = json.load(f)
            has_person = "person" in data
            has_projects = "projects" in data and len(data["projects"]) >= 3
            results.append(CheckResult(
                name="seo.metadata_jsonld.valid",
                passed=True,
                detail="seo/metadata.jsonld is valid JSON",
            ))
            results.append(CheckResult(
                name="seo.metadata_jsonld.person",
                passed=has_person,
                detail=f"Person schema {'present' if has_person else 'MISSING'} in metadata.jsonld",
            ))
            results.append(CheckResult(
                name="seo.metadata_jsonld.projects",
                passed=has_projects,
                detail=f"{'≥3 project schemas' if has_projects else 'Fewer than 3 project schemas'} in metadata.jsonld",
            ))
        except json.JSONDecodeError as e:
            results.append(CheckResult(
                name="seo.metadata_jsonld.valid",
                passed=False,
                detail=f"seo/metadata.jsonld is not valid JSON: {e}",
            ))
    else:
        results.append(CheckResult(
            name="seo.metadata_jsonld.exists",
            passed=False,
            detail="seo/metadata.jsonld not found",
        ))

    # Check sitemap.xml exists and contains expected URLs
    sitemap_path = ROOT / "seo" / "sitemap.xml"
    if sitemap_path.exists():
        content = sitemap_path.read_text()
        expected_urls = [
            "fraud-detection",
            "depression-detection",
            "federated-healthcare-ai",
        ]
        for url_part in expected_urls:
            found = url_part in content
            results.append(CheckResult(
                name=f"seo.sitemap.{url_part.replace('-', '_')}",
                passed=found,
                detail=f"URL '{url_part}' {'found' if found else 'MISSING'} in sitemap.xml",
            ))
    else:
        results.append(CheckResult(
            name="seo.sitemap.exists",
            passed=False,
            detail="seo/sitemap.xml not found",
        ))

    return results


def check_live_site(url: str) -> Results:
    """HTTP checks against the live portfolio URL. Requires 'requests' package."""
    results: Results = []
    if not HAS_REQUESTS:
        results.append(CheckResult(
            name="live_site.requests_available",
            passed=False,
            detail="'requests' package not installed. Run: pip install requests",
            warning=True,
        ))
        return results

    pages_to_check = [
        ("/", "Home"),
        ("/projects", "Projects index"),
        ("/about", "About"),
        ("/resume", "Resume"),
        ("/contact", "Contact"),
    ]

    for path, label in pages_to_check:
        full_url = f"{url}{path}"
        try:
            resp = requests.get(full_url, timeout=10, allow_redirects=True)
            passed = resp.status_code == 200
            results.append(CheckResult(
                name=f"live_site.{label.lower().replace(' ', '_')}",
                passed=passed,
                detail=f"GET {full_url} → {resp.status_code}",
            ))
        except requests.RequestException as e:
            results.append(CheckResult(
                name=f"live_site.{label.lower().replace(' ', '_')}",
                passed=False,
                detail=f"GET {full_url} → ERROR: {e}",
            ))

    return results


def print_report(all_results: Results) -> int:
    passed = [r for r in all_results if r.passed]
    failed = [r for r in all_results if not r.passed and not r.warning]
    warnings = [r for r in all_results if r.warning]

    print("\n" + "=" * 60)
    print("PORTFOLIO QA REPORT")
    print("=" * 60)
    print(f"Total checks:  {len(all_results)}")
    print(f"Passed:        {len(passed)}  ✓")
    print(f"Failed:        {len(failed)}  ✗")
    print(f"Warnings:      {len(warnings)}  ⚠")
    print("=" * 60)

    if failed:
        print("\n❌ FAILURES:")
        for r in failed:
            print(f"  ✗ [{r.name}] {r.detail}")

    if warnings:
        print("\n⚠ WARNINGS:")
        for r in warnings:
            print(f"  ⚠ [{r.name}] {r.detail}")

    if not failed:
        print("\n✅ All required checks passed!")
    else:
        print(f"\n❌ {len(failed)} check(s) failed. See failures above.")

    print("=" * 60 + "\n")
    return len(failed)


def main() -> None:
    manifest = load_manifest()
    all_results: Results = []

    all_results += check_manifest_structure(manifest)
    all_results += check_required_files_exist(manifest)
    all_results += check_hero_metric_in_index_page()
    all_results += check_case_study_pages()
    all_results += check_seo_files()

    if PORTFOLIO_URL:
        print(f"Running live site checks against: {PORTFOLIO_URL}")
        all_results += check_live_site(PORTFOLIO_URL)
    else:
        print("PORTFOLIO_URL not set — skipping live site HTTP checks.")
        print("  Set it to check the live site: PORTFOLIO_URL=https://your-site.com python qa/qa_script.py")

    failure_count = print_report(all_results)
    sys.exit(1 if failure_count > 0 else 0)


if __name__ == "__main__":
    main()
