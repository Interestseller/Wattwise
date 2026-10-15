"""Final integration test for WattWise Flask app."""
import sys
sys.path.insert(0, ".")
from run import create_app

app = create_app()
c = app.test_client()

errors = []

# ── Page routes
pages = ["/", "/analytics", "/forecasting", "/model-lab",
         "/bill-estimator", "/anomaly-monitor", "/settings"]
print("PAGES:")
for r in pages:
    resp = c.get(r)
    ok = b"WattWise" in resp.data
    tag = "OK" if resp.status_code == 200 and ok else "FAIL"
    print(f"  {tag}  {resp.status_code}  {r}")
    if tag == "FAIL":
        errors.append(f"Page {r} failed")

# ── API endpoints
print("APIS:")

r = c.get("/api/overview")
d = r.get_json()
usage = d.get("current_usage")
ok = r.status_code == 200 and usage is not None
print(f"  {'OK' if ok else 'FAIL'}  {r.status_code}  /api/overview  usage={usage}")
if not ok: errors.append("overview")

r = c.get("/api/analytics")
d = r.get_json()
pts = len(d.get("dates", []))
ok = r.status_code == 200 and pts > 0
print(f"  {'OK' if ok else 'FAIL'}  {r.status_code}  /api/analytics  points={pts}")
if not ok: errors.append("analytics")

r = c.get("/api/forecast?steps=24")
d = r.get_json()
pred = d.get("forecast", {}).get("predicted", [])
ok = r.status_code == 200 and len(pred) == 24
print(f"  {'OK' if ok else 'FAIL'}  {r.status_code}  /api/forecast  predicted={len(pred)}")
if not ok: errors.append("forecast")

r = c.get("/api/models")
d = r.get_json()
ok = r.status_code == 200 and "model_exists" in d
print(f"  {'OK' if ok else 'FAIL'}  {r.status_code}  /api/models  exists={d.get('model_exists')}")
if not ok: errors.append("models")

r = c.post("/api/bill-estimate",
           json={"tariff": 6.5, "fixed_charge": 50, "tax_rate": 0.05,
                 "reduction_pct": 0, "days": 30},
           content_type="application/json")
d = r.get_json()
total = d.get("total_bill")
ok = r.status_code == 200 and total is not None
print(f"  {'OK' if ok else 'FAIL'}  {r.status_code}  /api/bill-estimate  total={total}")
if not ok: errors.append("bill-estimate")

r = c.get("/api/anomalies?method=both&days=30")
d = r.get_json()
total = d.get("total")
ok = r.status_code == 200 and total is not None
print(f"  {'OK' if ok else 'FAIL'}  {r.status_code}  /api/anomalies  total={total}")
if not ok: errors.append("anomalies")

r = c.get("/api/export/csv")
ok = r.status_code == 200 and b"DateTime" in r.data
print(f"  {'OK' if ok else 'FAIL'}  {r.status_code}  /api/export/csv  bytes={len(r.data)}")
if not ok: errors.append("export/csv")

print()
if errors:
    print(f"FAILURES: {errors}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
