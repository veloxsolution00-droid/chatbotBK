import time
from app.rag.chat import answer

# Multi-questions based on the documents
multi_questions = [
    # Single question (should go through normal flow)
    'What is the concept of a deed?',
    
    # Multi-question: two questions in one
    'Who is Dracula? And who is Jonathan Harker?',
    
    # Multi-question: three questions
    'Who is Dracula? Who is Jonathan Harker? Who is Mina Murray?',
    
    # Big complex question about the legal document
    'What is the concept of a deed as discussed in the Legal_Documents_-Modernising_the_Formalities.pdf, and how does the definition evolve from Bracton (c. 1250) to modern law, and what are the key differences between a deed and a specialty, including the benefits and limitations of each?',
]

print("=" * 80)
print("MULTI-QUESTION & BIG QUESTION TEST")
print("=" * 80)

for q in multi_questions:
    start = time.perf_counter()
    result = answer(q)
    elapsed = time.perf_counter() - start
    
    print(f"\nQ: {q[:60]}{'...' if len(q) > 60 else ''}")
    print(f"  Status: {result['status']}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Question type: {result.get('question_type')}")
    print(f"  Citation valid: {result.get('citation_validation', {}).get('valid')}")
    
    # For multi-questions, show each answer
    if result.get('answers'):
        print(f"  Number of answers: {len(result['answers'])}")
        for i, ans in enumerate(result['answers']):
            ans_display = ans[:80] if ans and isinstance(ans, str) else str(ans)[:80] if ans else 'None'
            print(f"    Answer {i+1}: {ans_text}...")
    else:
        ans_text = result['answer'][:80] if result['answer'] else 'None'
        print(f"  Answer: {ans_text}...")
    
    # Show confidence
    conf = result.get('confidence', {})
    if conf:
        print(f"  Confidence: {conf.get('label', '?')} ({conf.get('score', '?')})")
    
    # Show sources
    sources = result.get('sources', [])
    print(f"  Sources: {len(sources)} chunks")
    
    # Show multi-question specific fields
    if result.get('question_resolved'):
        print(f"  Resolved from follow-up: {result['question_resolved']}")
    if result.get('question_plan'):
        plan = result['question_plan']
        ptype = plan.get('intent', '?')
        print(f"  Question plan: {ptype}")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)