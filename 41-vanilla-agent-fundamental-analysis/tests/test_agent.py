from fastapi.testclient import TestClient
from vanilla_agent_fundamental_analysis.main import app

def test_read_agent_config():
    """
    Ensure the agent exposes the correct configuration endpoint.
    This verifies that the app imports correctly and the route is registered.
    """
    client = TestClient(app)
    response = client.get("/agents.json")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check if our specific agent key exists in the response
    assert "vanilla_agent_fundamental_analysis" in data
    assert data["vanilla_agent_fundamental_analysis"]["name"] == "Fundamental Analyst"