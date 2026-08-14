from retriever import search_scheme


question = "Who is eligible for PM-KISAN?"

results = search_scheme(question)

for i, result in enumerate(results, start=1):
    print(f"\n===== RESULT {i} =====")
    print(result["content"])