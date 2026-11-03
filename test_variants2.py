from app.rag.chat import search as chat_search

results_deed = chat_search("what is deed", "local")
results_concept = chat_search("concept deed", "local")

print("=== what is deed top results ===")
for i, r in enumerate(results_deed[:10]):
    filename = r.get("metadata", {}).get("filename", "")
    location = r.get("metadata", {}).get("source_location", "")
    sim = r.get("similarity", 0)
    print(f"  {i+1}. {filename} {location} sim={sim}")

print("\n=== concept deed top results ===")
for i, r in enumerate(results_concept[:10]):
    filename = r.get("metadata", {}).get("filename", "")
    location = r.get("metadata", {}).get("source_location", "")
    sim = r.get("similarity", 0)
    print(f"  {i+1}. {filename} {location} sim={sim}")