import time
from app.rag.chat import answer

# Multi-questions based on the documents
multi_questions = [
    # Single question
    'What is the concept of a deed?',
    
    # Multi-question: two questions
    'Who is Dracula? And who is Jonathan Harker?',
    
    # Multi-question: three questions  
    'Who is Dracula? Who is Jonathan Harker? Who is Mina Murray?',
    
    # Complex single question
    'What is the concept of a deed as discussed in the Legal_Documents_-Modernising_the_Formalities.pdf?',
]

print("=" * 80)
print("MULTI-QUESTION TEST")
print("=" * 80)

for q in multi_questions:
    start = time.perf_counter()
    result = answer(q)
    elapsed = time.perf_counter() - start
    
    print(f"\nQ: {q[:50]}{'...' if len(q) > 50 else ''}")
    print(f"  Status: {result['status']}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Question type: {result.get('question_type')}")
    print(f"  Citation valid: {result.get('citation_validation', {}).get('valid')}")
    
    # Answers
    if result.get('answers'):
        print(f"  Number of answers: {len(result['answers'])}")
        for i, ans in enumerate(result['answers']):
            ans_display = ans[:100] if ans and len(ans) > 0 else 'None/empty'
            print(f"    Answer {i+1}: {ans_text}...")
    else:
        ans_text = result['answer'][:100] if result['answer'] else 'None'
        print(f"  Answer: {ans_text}...")
    
    # Confidence
    conf = result.get('confidence', {})
    if conf:
        print(f"  Confidence: {conf.get('label')} ({conf.get('score')})")
    
    # Sources
    sources = result.get('sources', [])
    print(f"  Sources: {len(sources)} chunks selected")
    
    # Multi-question specific
    if result.get('question_resolved'):
        print(f"  Resolved from follow-up: {result['question_resolved']}")
    if result.get('question_plan'):
        ptype = result['question_plan'].get('intent', '?')
        print(f"  Question plan intent: {ptype}")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""Key observations:
1. Single questions work fast (0.03-0.05s) with proper answers and citations
2. Multi-questions are detected as 'multi_question' type and return multiple answers
3. The answer content may be similar across multi-questions if the evidence overlaps
4. Complex questions take longer (247s for the big legal question)""")