from __future__ import annotations

import httpx

from french_territorial_intelligence.sources.registry import registry

BASE_URL = "https://geo.api.gouv.fr"
FIELDS = "nom,code,population,surface,codesPostaux,departement,centre"


class GeoSource:
    name = "geo"
    description = "French administrative geography (communes, departments)"

    async def fetch_territory(self, code_commune: str) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{BASE_URL}/communes/{code_commune}",
                params={"fields": FIELDS},
            )
            resp.raise_for_status()
            data = resp.json()
        return self._normalize(data)

    async def search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{BASE_URL}/communes",
                params={"nom": query, "fields": FIELDS, "boost": "population", "limit": 5},
            )
            resp.raise_for_status()
        return [self._normalize(c) for c in resp.json()]

    def available_metrics(self) -> list[str]:
        return ["population", "surface", "density", "department_code", "postal_codes"]

    @staticmethod
    def _normalize(data: dict) -> dict:
        pop = data.get("population", 0)
        surface_ha = data.get("surface", 0)
        surface_km2 = surface_ha / 100 if surface_ha else 0
        return {
            "name": data.get("nom", ""),
            "code": data.get("code", ""),
            "population": pop,
            "surface_km2": round(surface_km2, 2),
            "density": round(pop / surface_km2, 1) if surface_km2 > 0 else 0,
            "department_code": data.get("departement", {}).get("code", ""),
            "department_name": data.get("departement", {}).get("nom", ""),
            "postal_codes": data.get("codesPostaux", []),
            "coordinates": data.get("centre", {}).get("coordinates", []),
        }


registry.register(GeoSource())
