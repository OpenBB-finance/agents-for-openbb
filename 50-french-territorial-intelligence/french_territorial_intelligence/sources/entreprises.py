from __future__ import annotations

from collections import Counter

import httpx

from french_territorial_intelligence.sources.registry import registry

BASE_URL = "https://recherche-entreprises.api.gouv.fr/search"

NAF_SECTIONS = {
    "A": "Agriculture",
    "B": "Industries extractives",
    "C": "Manufacturing",
    "D": "Energy",
    "E": "Water & waste",
    "F": "Construction",
    "G": "Wholesale & retail",
    "H": "Transport",
    "I": "Hotels & restaurants",
    "J": "IT & communication",
    "K": "Finance & insurance",
    "L": "Real estate",
    "M": "Scientific & technical",
    "N": "Administrative services",
    "O": "Public administration",
    "P": "Education",
    "Q": "Health & social",
    "R": "Arts & entertainment",
    "S": "Other services",
}


def _build_div_to_section() -> dict[int, str]:
    """Build NAF Rev.2 division (2-digit) to section letter lookup."""
    mapping: dict[int, str] = {}
    for section, ranges in {
        "A": [(1, 3)],   "B": [(5, 9)],   "C": [(10, 33)],
        "D": [(35, 35)],  "E": [(36, 39)],  "F": [(41, 43)],
        "G": [(45, 47)],  "H": [(49, 53)],  "I": [(55, 56)],
        "J": [(58, 63)],  "K": [(64, 66)],  "L": [(68, 68)],
        "M": [(69, 75)],  "N": [(77, 82)],  "O": [(84, 84)],
        "P": [(85, 85)],  "Q": [(86, 88)],  "R": [(90, 93)],
        "S": [(94, 96)],
    }.items():
        for lo, hi in ranges:
            for div in range(lo, hi + 1):
                mapping[div] = section
    return mapping


_DIV_TO_SECTION = _build_div_to_section()


def _naf_to_section(naf_code: str) -> str:
    """Resolve a NAF activity code (e.g. '10.71A') to its section letter."""
    if not naf_code:
        return "?"
    # If the first character is already a section letter, use it directly
    if naf_code[0].isalpha():
        return naf_code[0].upper()
    # Otherwise extract the 2-digit division number
    digits = naf_code.replace(".", "")[:2]
    try:
        return _DIV_TO_SECTION.get(int(digits), "?")
    except ValueError:
        return "?"


class EntreprisesSource:
    name = "entreprises"
    description = "French business registry (SIRENE) — enterprise search"

    async def fetch_territory(self, code_commune: str) -> dict:
        department = code_commune[:2] if len(code_commune) == 5 else code_commune[:3]
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                BASE_URL,
                params={
                    "q": "",
                    "departement": department,
                    "per_page": 25,
                    "page": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        total = data.get("total_results", 0)
        return self._normalize(results, total, department)

    async def search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                BASE_URL,
                params={"q": query, "per_page": 10, "page": 1},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            {
                "siren": r.get("siren"),
                "name": r.get("nom_complet"),
                "sector": r.get("activite_principale"),
                "category": r.get("categorie_entreprise"),
                "created": r.get("date_creation"),
            }
            for r in data.get("results", [])
        ]

    def available_metrics(self) -> list[str]:
        return [
            "total_enterprises",
            "sector_breakdown",
            "category_breakdown",
            "enterprises",
        ]

    def _normalize(self, results: list[dict], total: int, department: str) -> dict:
        sectors: Counter[str] = Counter()
        categories: Counter[str] = Counter()
        enterprises = []
        for r in results:
            naf = r.get("activite_principale", "")
            section = _naf_to_section(naf)
            sector_name = NAF_SECTIONS.get(section, "Other")
            sectors[sector_name] += 1
            categories[r.get("categorie_entreprise", "Unknown")] += 1
            enterprises.append({
                "name": r.get("nom_complet", ""),
                "siren": r.get("siren", ""),
                "sector_code": naf,
                "sector": sector_name,
                "category": r.get("categorie_entreprise", ""),
                "open_establishments": r.get("nombre_etablissements_ouverts", 0),
                "created": r.get("date_creation", ""),
            })
        return {
            "department": department,
            "total_enterprises": total,
            "sector_breakdown": dict(sectors),
            "category_breakdown": dict(categories),
            "enterprises": enterprises,
        }


registry.register(EntreprisesSource())
