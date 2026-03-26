from __future__ import annotations

from collections import Counter

import httpx

from french_territorial_intelligence.sources.registry import registry

BASE_URL = "https://apidf-preprod.cerema.fr/dvf_opendata/mutations/"


class DvfSource:
    name = "dvf"
    description = "French real estate transactions (Demandes de Valeurs Foncieres)"

    async def fetch_territory(self, code_commune: str) -> dict:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                BASE_URL,
                params={
                    "code_insee": code_commune,
                    "anneemut_min": 2023,
                    "page_size": 100,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        total = data.get("count", 0)
        return self._normalize(results, total)

    async def search(self, query: str) -> list[dict]:
        return []

    def available_metrics(self) -> list[str]:
        return [
            "total_transactions",
            "avg_price_sqm",
            "median_price",
            "property_type_breakdown",
        ]

    @staticmethod
    def _normalize(results: list[dict], total: int) -> dict:
        prices_sqm = []
        prices = []
        types: Counter[str] = Counter()

        for r in results:
            price = float(r.get("valeurfonc") or 0)
            surface = float(r.get("sbati") or 0)
            prop_type = r.get("libtypbien", "Unknown")

            if price > 0:
                prices.append(price)
            if price > 0 and surface > 0:
                prices_sqm.append(price / surface)
            types[prop_type] += 1

        sorted_prices = sorted(prices)
        median = sorted_prices[len(sorted_prices) // 2] if sorted_prices else 0

        return {
            "total_transactions": total,
            "sample_size": len(results),
            "avg_price_sqm": round(sum(prices_sqm) / len(prices_sqm), 2) if prices_sqm else 0,
            "median_price": median,
            "min_price": min(prices) if prices else 0,
            "max_price": max(prices) if prices else 0,
            "property_type_breakdown": dict(types),
        }


registry.register(DvfSource())
