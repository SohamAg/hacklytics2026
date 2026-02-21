from backend_pipeline import HeartBackend

backend = HeartBackend("heart_features.csv")

result = backend.simulate_heart(["VSD", "SevereDilation"])

print(result)