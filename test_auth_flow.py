import sys
import types

if "chromadb.is_thin_client" not in sys.modules:
    _m = types.ModuleType("chromadb.is_thin_client")
    _m.is_thin_client = False
    sys.modules["chromadb.is_thin_client"] = _m

import datetime
import jwt
from fastapi.testclient import TestClient
from server import app, JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRATION_DAYS

client = TestClient(app)

def run_tests():
    test_user = f"cadet_{int(datetime.datetime.now().timestamp())}"
    test_pass = "securepass123"

    print("=== Test 1: Register with password mismatch ===")
    res = client.post("/api/auth/register", json={
        "username": test_user,
        "password": test_pass,
        "confirm_password": "different_password"
    })
    assert res.status_code == 400, f"Expected 400, got {res.status_code}: {res.text}"
    print("[+] Rejected mismatched passwords successfully:", res.json()["detail"])

    print("\n=== Test 2: Register with valid 2 inputs + 1 extra confirm password ===")
    res = client.post("/api/auth/register", json={
        "username": test_user,
        "password": test_pass,
        "confirm_password": test_pass
    })
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    token = data["token"]
    print(f"[+] Registration successful for {data['username']}, user_id: {data['user_id']}, IP: {data['client_ip']}")
    assert data["expires_in_days"] == 7, "Expected 7 days expiration"

    print("\n=== Test 3: Verify JWT Token Payload & > 5 Days Expiry ===")
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    exp_dt = datetime.datetime.fromtimestamp(payload["exp"], tz=datetime.timezone.utc)
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    delta_days = (exp_dt - now_dt).total_seconds() / 86400
    print(f"[+] Token exp: {exp_dt.isoformat()} (~{delta_days:.2f} days from now)")
    assert delta_days > 5.0, f"Expected token duration > 5 days, got {delta_days}"
    print(f"[+] Token bound IP: {payload.get('ip')}")
    assert payload.get("ip") == "127.0.0.1", f"Expected IP 127.0.0.1 in payload, got {payload.get('ip')}"

    print("\n=== Test 4: Login with strictly 2 inputs (username + password) ===")
    login_res = client.post("/api/auth/login", json={
        "username": test_user,
        "password": test_pass
    })
    assert login_res.status_code == 200, f"Expected 200, got {login_res.status_code}: {login_res.text}"
    login_data = login_res.json()
    login_token = login_data["token"]
    print(f"[+] Login successful with strictly 2 inputs. IP: {login_data['client_ip']}")

    print("\n=== Test 5: Access protected route (/api/auth/me) with matching IP ===")
    me_res = client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {login_token}"
    })
    assert me_res.status_code == 200, f"Expected 200, got {me_res.status_code}: {me_res.text}"
    me_data = me_res.json()
    print(f"[+] Authenticated profile accessed! User: {me_data['username']}, IP: {me_data['client_ip']}, Session: {me_data['session_expires_days']} days")

    print("\n=== Test 6: Access protected route with spoofed / mismatched IP (Security Check) ===")
    # Send request with a spoofed X-Forwarded-For header simulating a different network/client
    spoofed_res = client.get("/api/auth/me", headers={
        "Authorization": f"Bearer {login_token}",
        "x-forwarded-for": "198.51.100.42"
    })
    assert spoofed_res.status_code == 401, f"Expected 401 on mismatched IP, got {spoofed_res.status_code}: {spoofed_res.text}"
    print(f"[+] Rejected mismatched IP request as expected! Detail: {spoofed_res.json()['detail']}")

    print("\n[SUCCESS] All 6 Authentication & IP Binding & 7-Day Persistence tests passed!")

if __name__ == "__main__":
    run_tests()

