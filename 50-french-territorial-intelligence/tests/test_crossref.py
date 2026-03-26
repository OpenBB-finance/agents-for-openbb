from french_territorial_intelligence.crossref import (
    TerritoryProfile,
    build_profile,
    compare_profiles,
)


GEO_DATA = {
    "name": "Lyon",
    "code": "69123",
    "population": 519127,
    "surface_km2": 47.97,
    "density": 10817.1,
    "department_code": "69",
    "department_name": "Rhone",
    "postal_codes": ["69001"],
    "coordinates": [4.8351, 45.758],
}

ENT_DATA = {
    "department": "69",
    "total_enterprises": 45000,
    "sector_breakdown": {"IT & communication": 8, "Wholesale & retail": 6},
    "category_breakdown": {"PME": 15, "ETI": 5},
    "enterprises": [],
}

DVF_DATA = {
    "total_transactions": 1134,
    "sample_size": 100,
    "avg_price_sqm": 4763.58,
    "median_price": 350000,
    "min_price": 50000,
    "max_price": 2000000,
    "property_type_breakdown": {"UN APPARTEMENT": 80, "UNE MAISON": 20},
}


def test_build_profile():
    profile = build_profile(geo=GEO_DATA, entreprises=ENT_DATA, dvf=DVF_DATA)
    assert isinstance(profile, TerritoryProfile)
    assert profile.name == "Lyon"
    assert profile.population == 519127
    assert profile.enterprise_density > 0
    assert profile.affordability_index > 0


def test_enterprise_density_calculation():
    profile = build_profile(geo=GEO_DATA, entreprises=ENT_DATA, dvf=DVF_DATA)
    # 45000 / 519127 * 1000 = 86.69
    assert 86 < profile.enterprise_density < 87


def test_build_profile_missing_dvf():
    profile = build_profile(geo=GEO_DATA, entreprises=ENT_DATA, dvf=None)
    assert profile.avg_price_sqm == 0
    assert profile.affordability_index == 0
    assert profile.enterprise_density > 0


def test_build_profile_missing_entreprises():
    profile = build_profile(geo=GEO_DATA, entreprises=None, dvf=DVF_DATA)
    assert profile.total_enterprises == 0
    assert profile.enterprise_density == 0
    assert profile.avg_price_sqm > 0


def test_insights_generated():
    profile = build_profile(geo=GEO_DATA, entreprises=ENT_DATA, dvf=DVF_DATA)
    assert len(profile.insights) > 0
    # Should have at least one insight about enterprise density and one about sectors
    all_insights = " ".join(profile.insights).lower()
    assert "enterprise" in all_insights or "density" in all_insights or "sector" in all_insights


def test_compare_profiles():
    p1 = build_profile(geo=GEO_DATA, entreprises=ENT_DATA, dvf=DVF_DATA)

    geo2 = {**GEO_DATA, "name": "Nantes", "code": "44109", "population": 320732, "density": 4600}
    ent2 = {**ENT_DATA, "total_enterprises": 28000}
    dvf2 = {**DVF_DATA, "avg_price_sqm": 3500}
    p2 = build_profile(geo=geo2, entreprises=ent2, dvf=dvf2)

    comparison = compare_profiles(p1, p2)
    assert "advantages" in comparison
    # Lyon should have higher enterprise density, Nantes lower price
    lyon_adv = comparison["advantages"].get("Lyon", [])
    nantes_adv = comparison["advantages"].get("Nantes", [])
    assert len(lyon_adv) + len(nantes_adv) > 0


def test_compare_profiles_with_missing_data():
    p1 = build_profile(geo=GEO_DATA, entreprises=ENT_DATA, dvf=None)
    geo2 = {**GEO_DATA, "name": "Nantes", "code": "44109", "population": 320732}
    p2 = build_profile(geo=geo2, entreprises=None, dvf=None)
    comparison = compare_profiles(p1, p2)
    assert "advantages" in comparison
