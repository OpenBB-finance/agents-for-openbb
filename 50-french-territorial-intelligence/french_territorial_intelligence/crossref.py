"""CrossRef Engine — cross-reference multiple territorial data sources.

Combines geo, enterprise, and real-estate data into a unified
TerritoryProfile with derived indicators and human-readable insights.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TerritoryProfile:
    """Cross-referenced territorial profile from multiple data sources."""

    name: str
    code: str
    population: int
    surface_km2: float
    density: float
    department: str

    # Enterprise metrics
    total_enterprises: int = 0
    enterprise_density: float = 0  # per 1000 inhabitants
    top_sectors: dict[str, int] = field(default_factory=dict)
    enterprise_categories: dict[str, int] = field(default_factory=dict)

    # Real estate metrics
    avg_price_sqm: float = 0
    median_price: float = 0
    total_transactions: int = 0
    property_types: dict[str, int] = field(default_factory=dict)

    # Cross-referenced indicators
    affordability_index: float = 0  # price_sqm / enterprise_density
    economic_intensity: float = 0   # enterprises * density / population
    insights: list[str] = field(default_factory=list)


def build_profile(
    geo: dict,
    entreprises: dict | None = None,
    dvf: dict | None = None,
) -> TerritoryProfile:
    """Build a cross-referenced territory profile from raw source data."""
    pop = geo.get("population", 0)

    profile = TerritoryProfile(
        name=geo.get("name", ""),
        code=geo.get("code", ""),
        population=pop,
        surface_km2=geo.get("surface_km2", 0),
        density=geo.get("density", 0),
        department=geo.get("department_name", ""),
    )

    if entreprises:
        total_ent = entreprises.get("total_enterprises", 0)
        profile.total_enterprises = total_ent
        profile.enterprise_density = round(total_ent / pop * 1000, 2) if pop > 0 else 0
        profile.top_sectors = entreprises.get("sector_breakdown", {})
        profile.enterprise_categories = entreprises.get("category_breakdown", {})

    if dvf:
        profile.avg_price_sqm = dvf.get("avg_price_sqm", 0)
        profile.median_price = dvf.get("median_price", 0)
        profile.total_transactions = dvf.get("total_transactions", 0)
        profile.property_types = dvf.get("property_type_breakdown", {})

    _compute_cross_indicators(profile)
    _generate_insights(profile)
    return profile


def _compute_cross_indicators(profile: TerritoryProfile) -> None:
    """Derive indicators from cross-referencing multiple sources."""
    if profile.enterprise_density > 0 and profile.avg_price_sqm > 0:
        profile.affordability_index = round(
            profile.avg_price_sqm / profile.enterprise_density, 2
        )
    if profile.total_enterprises > 0 and profile.population > 0:
        profile.economic_intensity = round(
            (profile.total_enterprises * profile.density) / profile.population, 2
        )


def _generate_insights(profile: TerritoryProfile) -> None:
    """Generate human-readable insights from cross-referenced data."""
    insights = []

    if profile.enterprise_density > 100:
        insights.append(
            f"High enterprise density ({profile.enterprise_density:.0f} per 1,000 inhabitants) "
            f"indicates a mature economic ecosystem."
        )
    elif profile.enterprise_density > 50:
        insights.append(
            f"Moderate enterprise density ({profile.enterprise_density:.0f} per 1,000 inhabitants)."
        )
    elif profile.enterprise_density > 0:
        insights.append(
            f"Low enterprise density ({profile.enterprise_density:.0f} per 1,000 inhabitants) "
            f"— room for economic development."
        )

    if profile.avg_price_sqm > 0 and profile.enterprise_density > 0:
        if profile.affordability_index < 30:
            insights.append(
                "Low affordability index — real estate costs are well-balanced "
                "relative to economic activity."
            )
        elif profile.affordability_index > 80:
            insights.append(
                "High affordability index — real estate costs may be a barrier "
                "for new business establishment."
            )
        else:
            insights.append(
                f"Affordability index of {profile.affordability_index:.0f} — "
                f"moderate real estate cost relative to economic activity."
            )

    if profile.top_sectors:
        top = sorted(profile.top_sectors.items(), key=lambda x: x[1], reverse=True)[:3]
        sectors_str = ", ".join(f"{s} ({n})" for s, n in top)
        insights.append(f"Leading sectors: {sectors_str}.")

    if profile.avg_price_sqm > 0:
        insights.append(f"Average real estate price: {profile.avg_price_sqm:,.0f} EUR/sqm.")

    profile.insights = insights


def compare_profiles(
    profile_a: TerritoryProfile,
    profile_b: TerritoryProfile,
) -> dict:
    """Compare two territory profiles and highlight key differences."""
    advantages: dict[str, list[str]] = {profile_a.name: [], profile_b.name: []}

    comparisons = [
        ("enterprise_density", "higher enterprise density", True),
        ("avg_price_sqm", "lower real estate cost", False),
        ("population", "larger population", True),
        ("affordability_index", "better affordability ratio", False),
        ("total_transactions", "more active real estate market", True),
    ]

    for attr, label, higher_is_better in comparisons:
        val_a = getattr(profile_a, attr, 0)
        val_b = getattr(profile_b, attr, 0)
        if val_a == 0 or val_b == 0:
            continue
        if higher_is_better:
            winner = profile_a.name if val_a > val_b else profile_b.name
        else:
            winner = profile_a.name if val_a < val_b else profile_b.name
        pct = abs(val_a - val_b) / max(val_a, val_b) * 100
        advantages[winner].append(f"{label} ({pct:.0f}% difference)")

    return {"advantages": advantages}
