from app.rag.chat import search as chat_search

results_deed = chat_search("what is deed", "local")
results_concept = chat_search("concept deed", "local")
results_concept_of_deed = chat_search("what is concept of deed", "local")

print("Search \"what is deed\" returns", len(results_deed), "chunks")
print("Search \"concept deed\" returns", len(results_concept), "chunks")  
print("Search \"what is concept of deed\" returns", len(results_concept_of_deed), "chunks")

if results_deed:
    print("\nFirst \"what is deed\" result:", results_deed[0].get("metadata", {}).get("filename"), results_deed[0].get("similarity"))
if results_concept:
    print("First \"concept deed\" result:", results_concept[0].get("metadata", {}).get("filename"), results_concept[0].get("similarity"))
if results_concept_of_deed:
    print("First \"what is concept of deed\" result:", results_concept_of_deed[0].get("metadata", {}).get("filename"), results_concept_of_deed[0].get("similarity"))