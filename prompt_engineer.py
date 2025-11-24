import json
import pandas as pd


# --------------------------------------------------------------------------
# Helper: Get Examples
# --------------------------------------------------------------------------
def get_few_shot_examples(job, example_banks):
    """
    Retrieves 2-3 examples from the CSV based on CEFR and Type.
    Robustly handles column name mismatches (e.g., extra spaces).
    """
    bank = example_banks.get(job['type'].lower())
    if bank is None or bank.empty: 
        return ""
    
    bank.columns = [c.strip() for c in bank.columns]
    
    if 'CEFR rating' in bank.columns:
        relevant = bank[bank['CEFR rating'].astype(str).str.strip() == str(job['cefr']).strip()]
    else:
        relevant = bank

    if len(relevant) >= 2:
        samples = relevant.sample(2)
    elif len(bank) >= 2:
        samples = bank.sample(2) 
    else:
        return "" 

    output = ""
    for _, row in samples.iterrows():
        ex_dict = {
            "Question Prompt": row.get("Question Prompt", "N/A"),
            "Answer A": row.get("Answer A", "N/A"),
            "Answer B": row.get("Answer B", "N/A"),
            "Answer C": row.get("Answer C", "N/A"),
            "Answer D": row.get("Answer D", "N/A"),
            "Correct Answer": row.get("Correct Answer", "N/A")
        }
        output += "### EXAMPLE:\n" + json.dumps(ex_dict) + "\n\n"
    return output


# --------------------------------------------------------------------------
# Strategy: Sequential BATCH MODE - THREE-STAGE ARCHITECTURE (MINIMAL FIXES)
# --------------------------------------------------------------------------

def create_sequential_batch_stage1_prompt(job_list, example_banks):
    """
    STAGE ONE: Generates complete sentences with correct answers and context clues for ALL jobs at once.
    Includes multi-word phrase splitting strategy and distinguishes between 
    grammatical versus semantic constraint requirements.
    """
    examples = get_few_shot_examples(job_list[0], example_banks) if job_list else ""
    
    system_msg = f"""You are an expert ELT content creator. You will generate exactly {len(job_list)} complete test questions in a single JSON response. 

CRITICAL: Your entire response must be a JSON object with a "questions" key containing an array of exactly {len(job_list)} question objects. Do not generate fewer questions than requested."""
    
    # Build the batch specification
    job_specs = []
    has_grammar_distinction = False
    has_vocabulary = False
    
    for job in job_list:
        raw_context = job.get('context', 'General')
        main_topic = raw_context
        
        job_specs.append({
            "job_id": job['job_id'],
            "cefr": job['cefr'],
            "type": job['type'],
            "focus": job['focus'],
            "topic": main_topic
        })
        
        if job['type'] == 'Grammar' and 'vs' in job['focus'].lower():
            has_grammar_distinction = True
        if job['type'] == 'Vocabulary':
            has_vocabulary = True
    
    # Determine appropriate constraint instruction
    constraint_instruction = ""
    
    if has_grammar_distinction:
        constraint_instruction += """
GRAMMATICAL EXCLUSIVITY RULE (for grammar distinction questions):
When the Assessment Focus contains "vs" (e.g., "going to vs will", "Past Simple vs Present Perfect"), 
the Complete Sentence MUST include a GRAMMATICAL SIGNAL that makes only the correct answer structurally valid.

Examples of grammatical signals:
- Time markers: "yesterday" forces Past Simple, eliminates Present Perfect
- Evidence phrases: "Look at those clouds" forces "going to", eliminates "will"
- Hypothetical markers: "If I were you" forces Type 2 conditional, eliminates Type 1
- Duration markers: "for five years" forces Present Perfect, eliminates Past Simple
- Definiteness markers: "already" forces Present Perfect, eliminates Past Simple

The distractors should be GRAMMATICALLY INCOMPATIBLE with the sentence structure, not merely semantically weaker.
"""
    
    if has_vocabulary:
        constraint_instruction += """
SEMANTIC EXCLUSIVITY RULE (for vocabulary questions):
The Complete Sentence must contain SEMANTIC CONTEXT CLUES that make only the correct answer logically appropriate.

Context clue strategies by level:
- A1-A2: Category membership, clear antonyms, basic collocations (verb-noun pairings)
- B1-B2: Connotation distinctions, phrasal verb meanings, word form requirements, collocation violations
- C1: Precise semantic distinctions, idiomatic expressions, academic collocations

The distractors should be SEMANTICALLY INCOMPATIBLE or IDIOMATICALLY WRONG with the context, even if grammatically valid.
"""
    
    user_msg = f"""
TASK: Create exactly {len(job_list)} complete, original test questions from scratch.

You must generate ALL {len(job_list)} questions in this single response. Each question specification below MUST have a corresponding question in your output.

JOB SPECIFICATIONS (one question for each):
{json.dumps(job_specs, indent=2)}

{constraint_instruction}

GENERATION INSTRUCTIONS FOR EACH QUESTION:

1. **ANTI-REPETITION (CRITICAL):** Each question must have a UNIQUE topic and scenario. Do NOT reuse themes, contexts, or vocabulary across questions.

2. **INTEGRATED CONSTRUCTION - GRAMMAR QUESTIONS:** For multi-word grammatical constructions being tested (such as "going to", "have to", "used to"), strategically position elements to create structural constraints. If testing "going to" versus "will", consider placing auxiliary verbs in the stem such as "It's _____ rain" where the contracted auxiliary eliminates "will" structurally. Multi-word answers should be separated across stem and answer slot when this creates grammatical enforcement.

3. **INTEGRATED CONSTRUCTION - VOCABULARY QUESTIONS:** Place the target vocabulary word within an authentic sentence that provides semantic context clues. Ensure the target word is at an appropriate lexical level for the CEFR rating. For higher-level vocabulary items, the sentence context must make the meaning clear enough that learners can discriminate it from phonetically similar alternatives.

4. **CONTEXT CLUE ENGINEERING - GRAMMAR QUESTIONS:** Include grammatical signals that structurally eliminate incorrect options. Time markers such as "yesterday" or "for five years", structural elements such as auxiliary verb placement, or syntactic requirements such as "enjoy" requiring a gerund create grammatical incompatibility rather than semantic implausibility. The context clue must make distractors grammatically wrong, not just semantically odd.

5. **CONTEXT CLUE ENGINEERING - VOCABULARY QUESTIONS:** Include semantic context clues that make only the correct answer logically appropriate while keeping all options grammatically valid. For verb collocations, ensure the full sentence structure eliminates incorrect verbs through constructions such as benefactive phrases. The context clue must make distractors semantically incompatible while remaining grammatically acceptable.

6. **METALINGUISTIC REFLECTION (REQUIRED):** Explicitly identify which portion functions as the context clue and explain whether it creates grammatical elimination or semantic elimination based on question type.

7. **NEGATIVE CONSTRAINT (VERBOSITY):** Sentences must be concise (max 2 sentences). No preambles. Do NOT use imperative commands.

8. **NEGATIVE CONSTRAINT (METALANGUAGE):** NEVER use grammar terminology in the sentence itself.

9. **LOGICAL COHERENCE CHECK:** Review your complete sentence to ensure it is semantically coherent and factually plausible. Avoid nonsensical combinations such as "The meeting was cancelled so we put it off until next month" where the actions contradict each other.

MANDATORY OUTPUT FORMAT:
{{
  "questions": [
    {{
      "Item Number": "...",
      "Assessment Focus": "...",
      "Complete Sentence": "...[sentence with answer visible]...",
      "Correct Answer": "...",
      "Context Clue Location": "...[which phrase/clause]...",
      "Context Clue Explanation": "...[why this eliminates alternatives]...",
      "CEFR rating": "...",
      "Category": "..."
    }},
    {{
      "Item Number": "...",
      "Assessment Focus": "...",
      "Complete Sentence": "...",
      "Correct Answer": "...",
      "Context Clue Location": "...",
      "Context Clue Explanation": "...",
      "CEFR rating": "...",
      "Category": "..."
    }}
    ... (continue until you have exactly {len(job_list)} question objects)
  ]
}}

VERIFICATION: Count your question objects before submitting. You must have exactly {len(job_list)} items in the "questions" array.

STYLE REFERENCE (format guide only - do not copy scenarios):
{examples}
"""
    return system_msg, user_msg


def create_sequential_batch_stage2_grammar_prompt(job_list, stage1_outputs):
    """
    STAGE TWO (GRAMMAR): Generates candidate distractor pools for grammar questions.
    MINIMAL FIXES: Lexical overlap prohibition, target form coverage, proficiency-appropriate errors.
    """
    system_msg = f"""You are an expert ELT test designer specializing in grammar assessment. You will generate candidate distractors for exactly {len(job_list)} grammar questions in a single JSON response with a "candidates" key."""
    
    user_msg = f"""
TASK: Generate 5 candidate distractors for ALL {len(job_list)} GRAMMAR questions.

INPUT FROM STAGE 1 (Complete sentences with correct answers):
{json.dumps(stage1_outputs, indent=2)}

GENERATION INSTRUCTIONS:

For EACH question, generate 5 alternative options that replace the Correct Answer in the Complete Sentence. Your alternatives must render the sentence grammatically incorrect, but should be plausible errors that intermediate learners (CEFR A2-B2) might produce.

CONSTRAINTS:

1. **WORD COUNT LIMIT:** Each candidate must be MAXIMUM 3 words.

2. **GRAMMATICAL PARALLELISM:** All candidates must match the word count and construction type of the correct answer. If the correct answer is two words, candidates should be two words.

3. **LEXICAL OVERLAP PROHIBITION:** Do NOT use any words that appear AFTER the blank in the Complete Sentence. Example: "He ____ have won the race" → do not use "have" or "won" in candidates. This prevents nonsensical repetition.

4. **TARGET FORM COVERAGE:** For Assessment Focus containing "vs" (e.g., "Gerunds & Infinitives", "going to vs will"), candidates MUST include examples of BOTH forms being tested. Example: if testing gerunds vs infinitives, include both gerund forms AND infinitive forms in your 5 candidates.

5. **PROFICIENCY-APPROPRIATE ERRORS:** Match error sophistication to target CEFR level. Advanced questions (B2+) should have subtle form confusions within complex structures, not elementary violations. Lower levels (A1-B1) can use basic form errors.

6. **PLAUSIBLE LEARNER ERRORS:** Generate common interlanguage patterns such as:
   - Missing infinitive markers (e.g., "is going rain" instead of "is going to rain")
   - Incorrect auxiliary combinations (e.g., "will be rain" instead of "will rain")
   - Subject-verb agreement violations (e.g., "he go" instead of "he goes")
   - Tense confusion with similar forms (e.g., "has went" instead of "has gone")
   - Incorrect inflections (e.g., "will rains" instead of "will rain")

7. **INFLECTIONS AND DERIVATIONS PERMITTED:** You may modify verb forms, add or remove inflections, and adjust derivations to create grammatically incorrect constructions.

8. **NO LEXICAL OVERLAP WITH CORRECT ANSWER:** Do not use any form of the correct answer word or its root in candidates unless testing word form distinctions.

9. **ANTI-REPETITION:** Avoid using identical candidate words across multiple questions in this batch unless required by the Assessment Focus.

MANDATORY OUTPUT FORMAT:
{{
  "candidates": [
    {{
      "Item Number": "...",
      "Candidate A": "...[max 3 words]...",
      "Candidate B": "...[max 3 words]...",
      "Candidate C": "...[max 3 words]...",
      "Candidate D": "...[max 3 words]...",
      "Candidate E": "...[max 3 words]..."
    }},
    ... (exactly {len(job_list)} candidate sets)
  ]
}}

VERIFICATION: You must generate exactly {len(job_list)} candidate sets with 5 candidates each.
"""
    return system_msg, user_msg


def create_sequential_batch_stage2_vocabulary_prompt(job_list, stage1_outputs):
    """
    STAGE TWO (VOCABULARY): Generates candidate distractor pools for vocabulary questions.
    MINIMAL FIX: Exact inflectional form matching with concrete examples.
    """
    system_msg = f"""You are an expert ELT test designer specializing in vocabulary assessment. You will generate candidate distractors for exactly {len(job_list)} vocabulary questions in a single JSON response with a "candidates" key."""
    
    user_msg = f"""
TASK: Generate 5 candidate distractors for ALL {len(job_list)} VOCABULARY questions.

INPUT FROM STAGE 1 (Complete sentences with correct answers):
{json.dumps(stage1_outputs, indent=2)}

GENERATION INSTRUCTIONS:

For EACH question, generate 5 alternative options that replace the Correct Answer in the Complete Sentence. Your alternatives must render the sentence contextually inappropriate through collocational mismatch, idiomatic violation, or register conflict, while maintaining grammatical correctness.

CONSTRAINTS:

1. **WORD COUNT LIMIT:** Each candidate must be MAXIMUM 3 words.

2. **EXACT INFLECTIONAL FORM MATCHING:** ALL candidates must match the PRECISE grammatical form of the correct answer - same tense, aspect, person, number, and voice. Examples:
   - Correct: "brush" (base) → Candidates: "comb", "wash", "scrub" (NOT "combs", "washes", "scrubs")
   - Correct: "brushes" (3rd singular) → Candidates: "combs", "washes", "scrubs" (NOT "comb", "wash", "scrub")
   - Correct: "plowed through" (past) → Candidates: "broke through", "pushed through" (NOT "break through", "push through")
   **FORM IDENTIFICATION RULE:** Before generating candidates, check: Does the correct answer follow 'to', a modal (can/will/should/want to), or a plural subject (I/you/we/they)? If YES, it is BASE FORM - all candidates must be base form with NO -s/-es/-ed endings.
   

3. **SAME WORD CLASS AND GRAMMATICAL FORM REQUIRED:** All candidates must match the exact grammatical structure of the correct answer. If the correct answer is a present tense third person singular verb, all candidates must be present tense third person singular verbs.

4. **SEMANTIC FIELD PROXIMITY:** Generate alternatives from the SAME semantic field as the correct answer. For example:
   - If the correct answer is "take a look at" (visual observation), candidates should be other observation phrases like "behold", "gaze upon", "pay heed to"
   - If the correct answer is "postpone" (delay action), candidates should be other temporal or planning verbs like "possess", "position", "compose"

5. **CONTEXTUAL INAPPROPRIATENESS TYPES:**
   - **Register Conflict:** Formal/archaic terms in casual contexts (e.g., "behold" for "look at")
   - **Collocational Mismatch:** Verbs that don't combine naturally with following nouns (e.g., "make breakfast" vs "do breakfast")
   - **Idiomatic Violation:** Phrases that violate standard usage patterns (e.g., "pay heed to" with visual objects)

6. **PHONETIC AND SEMANTIC ALTERNATIVES PERMITTED:** For higher-level vocabulary (B1+), include at least one phonetically similar alternative that matches lexical sophistication (e.g., "possess" or "position" as alternatives to "postpone").

7. **NO LEXICAL OVERLAP:** Do not use any form of the correct answer word or its root in candidates.

8. **ANTI-REPETITION:** Avoid using identical candidate words across multiple questions in this batch.

MANDATORY OUTPUT FORMAT:
{{
  "candidates": [
    {{
      "Item Number": "...",
      "Candidate A": "...[max 3 words]...",
      "Candidate B": "...[max 3 words]...",
      "Candidate C": "...[max 3 words]...",
      "Candidate D": "...[max 3 words]...",
      "Candidate E": "...[max 3 words]..."
    }},
    ... (exactly {len(job_list)} candidate sets)
  ]
}}

VERIFICATION: You must generate exactly {len(job_list)} candidate sets with 5 candidates each.
"""
    return system_msg, user_msg


def create_sequential_batch_stage3_grammar_prompt(job_list, stage1_outputs, stage2_outputs):
    """
    STAGE THREE (GRAMMAR): Binary validation with filtering authority.
    MINIMAL FIX: Proficiency-appropriate error checking with rejection of too-easy errors.
    """
    system_msg = f"""You are an expert English grammar validator. You will evaluate candidate distractors for exactly {len(job_list)} grammar questions and return your validated selections in a JSON object with a "validated" key."""
    
    # Construct validation input combining Stage 1 and Stage 2 data
    validation_input = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        complete_sentence = s1.get("Complete Sentence", "")
        correct_answer = s1.get("Correct Answer", "")
        
        validation_input.append({
            "Item Number": s1.get("Item Number", ""),
            "Assessment Focus": s1.get("Assessment Focus", ""),
            "Complete Sentence": complete_sentence,
            "Correct Answer": correct_answer,
            "Candidate A": s2.get("Candidate A", ""),
            "Candidate B": s2.get("Candidate B", ""),
            "Candidate C": s2.get("Candidate C", ""),
            "Candidate D": s2.get("Candidate D", ""),
            "Candidate E": s2.get("Candidate E", ""),
            "CEFR": job['cefr']
        })
    
    user_msg = f"""
TASK: Validate candidate distractors for ALL {len(job_list)} GRAMMAR questions and select the final three distractors per question.

VALIDATION INPUT:
{json.dumps(validation_input, indent=2)}

VALIDATION PROCEDURE:

For EACH question, you must test ALL FIVE candidates by performing the following steps:

1. **SENTENCE RECONSTRUCTION:** Take the Complete Sentence and replace the Correct Answer with each candidate individually.

2. **GRAMMATICAL CORRECTNESS TEST:** Evaluate whether the resulting sentence is grammatically correct in standard English. Ask yourself: "Does this sentence obey English grammatical rules regardless of whether it makes contextual sense?"

3. **PROFICIENCY-APPROPRIATE ERROR CHECK:** For each candidate, ask: "Does this error type match the CEFR level being tested?" Reject candidates with elementary errors (e.g., "didn't needed", "hasn't to", "must to") when testing B2+ constructions. These are too easy and fail to assess the target skill.

4. **BINARY CLASSIFICATION:** For each candidate, answer YES or NO:
   - YES = The sentence is grammatically correct (structural rules are followed)
   - NO = The sentence is grammatically incorrect (structural rules are violated)

5. **FILTER AND SELECT:** 
   - REJECT all candidates that produce grammatically correct sentences (YES answers)
   - REJECT all candidates with proficiency mismatches (too-easy errors for the level)
   - RETAIN all candidates that produce grammatically incorrect sentences at appropriate proficiency level
   - SELECT the best THREE candidates from the retained pool based on:
     * Plausibility as learner errors
     * Diversity of error types
     * Alignment with Assessment Focus (for distinction questions)

CRITICAL DISTINCTION:

You are testing GRAMMATICAL CORRECTNESS, not contextual appropriateness. A sentence like "It will rain tomorrow" is grammatically correct even if the context suggests present continuous would be more natural. A sentence like "It will be rain tomorrow" is grammatically incorrect because the structure violates English grammar rules.

Examples:
- "Look at those clouds, it will rain" → GRAMMATICALLY CORRECT (appropriate tense structure)
- "Look at those clouds, it will be rain" → GRAMMATICALLY INCORRECT (malformed future construction)
- "Look at those clouds, it has rained" → GRAMMATICALLY CORRECT (appropriate tense structure)
- "Look at those clouds, it is going rain" → GRAMMATICALLY INCORRECT (missing infinitive marker)

PROFICIENCY CHECK EXAMPLES:
- C1 modal question + "didn't needed to do" distractor → REJECT (A2-level error, too easy for C1)
- C1 modal question + "couldn't have done" distractor → ACCEPT (appropriate C1-level confusion)
- B2 conditional + "If I will know" distractor → REJECT (A2-level error, too easy for B2)
- A2 past tense + "goed" distractor → ACCEPT (appropriate A2-level error)

MANDATORY OUTPUT FORMAT:
{{
  "validated": [
    {{
      "Item Number": "...",
      "Selected Distractor A": "...",
      "Selected Distractor B": "...",
      "Selected Distractor C": "...",
      "Validation Notes": "Brief explanation of which candidates passed/failed and why"
    }},
    ... (exactly {len(job_list)} validated sets)
  ]
}}

VERIFICATION: You must provide exactly {len(job_list)} validated distractor sets with exactly 3 selected distractors each.
"""
    return system_msg, user_msg


def create_sequential_batch_stage3_vocabulary_prompt(job_list, stage1_outputs, stage2_outputs):
    """
    STAGE THREE (VOCABULARY): Binary validation with filtering authority.
    MINIMAL FIX: Native speaker acceptability check to reject multiple correct answers.
    """
    system_msg = f"""You are an expert English vocabulary validator. You will evaluate candidate distractors for exactly {len(job_list)} vocabulary questions and return your validated selections in a JSON object with a "validated" key."""
    
    # Construct validation input combining Stage 1 and Stage 2 data
    validation_input = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        complete_sentence = s1.get("Complete Sentence", "")
        correct_answer = s1.get("Correct Answer", "")
        
        validation_input.append({
            "Item Number": s1.get("Item Number", ""),
            "Assessment Focus": s1.get("Assessment Focus", ""),
            "Complete Sentence": complete_sentence,
            "Correct Answer": correct_answer,
            "Candidate A": s2.get("Candidate A", ""),
            "Candidate B": s2.get("Candidate B", ""),
            "Candidate C": s2.get("Candidate C", ""),
            "Candidate D": s2.get("Candidate D", ""),
            "Candidate E": s2.get("Candidate E", ""),
            "CEFR": job['cefr']
        })
    
    user_msg = f"""
TASK: Validate candidate distractors for ALL {len(job_list)} VOCABULARY questions and select the final three distractors per question.

VALIDATION INPUT:
{json.dumps(validation_input, indent=2)}

VALIDATION PROCEDURE:

For EACH question, you must test ALL FIVE candidates by performing the following steps:

1. **SENTENCE RECONSTRUCTION:** Take the Complete Sentence and replace the Correct Answer with each candidate individually.

2. **DUAL VALIDATION TEST:** For each reconstructed sentence, answer TWO questions:
   
   Question A: Is the sentence grammatically correct in standard English?
   - YES = The sentence follows all structural rules
   - NO = The sentence violates grammatical rules
   
   Question B: Is the sentence semantically appropriate in this context?
   - YES = The sentence makes logical sense and uses natural English
   - NO = The sentence contains collocational errors, register mismatches, or idiomatic violations

3. **UNIQUENESS CHECK:** For candidates that pass A and B, ask: "If an examiner saw BOTH the correct answer AND this candidate as student responses, would both receive full marks?" If YES, reject the candidate - the test has multiple correct answers. Only the intended answer should be fully acceptable.

4. **FILTERING CRITERIA:**
   - REJECT candidates that answer NO to Question A (grammatically incorrect)
   - REJECT candidates that answer YES to Question B (semantically appropriate)
   - REJECT candidates that native speakers would accept (multiple correct answers)
   - RETAIN only candidates that are grammatically correct BUT semantically/idiomatically wrong

5. **SELECT FINAL THREE:**
   - From the retained pool, SELECT the best THREE candidates based on:
     * Plausibility as vocabulary confusions
     * Quality of the contextual inappropriateness (register conflict, collocation error, idiom violation)
     * Diversity of error types
     * Lexical level matching for higher-level target words

CRITICAL DISTINCTION:

You are testing for grammatical correctness COMBINED WITH semantic inappropriateness. The ideal vocabulary distractor produces a grammatically perfect sentence that is contextually wrong.

Examples:
- "Take a look at the clouds" → CORRECT (grammatically correct, semantically appropriate)
- "Behold the clouds" → VALID DISTRACTOR (grammatically correct, register conflict makes it inappropriate)
- "Gaze upon the clouds" → VALID DISTRACTOR (grammatically correct, collocational mismatch for casual context)
- "Look to the clouds" → INVALID (grammatically questionable, missing proper preposition)
- "Observe at the clouds" → INVALID (grammatically incorrect, wrong preposition after "observe")

NATIVE SPEAKER ACCEPTABILITY EXAMPLES:
- "managed to scrape together money" when correct answer is "squirrel away" → REJECT (both acceptable)
- "likes to mess with engines" when correct answer is "tinker with" → REJECT (both acceptable colloquially)
- "managed to pile up money" when correct answer is "squirrel away" → ACCEPT (collocational mismatch)

MANDATORY OUTPUT FORMAT:
{{
  "validated": [
    {{
      "Item Number": "...",
      "Selected Distractor A": "...",
      "Selected Distractor B": "...",
      "Selected Distractor C": "...",
      "Validation Notes": "Brief explanation of which candidates passed/failed and why"
    }},
    ... (exactly {len(job_list)} validated sets)
  ]
}}

VERIFICATION: You must provide exactly {len(job_list)} validated distractor sets with exactly 3 selected distractors each.
"""
    return system_msg, user_msg


# --------------------------------------------------------------------------
# Legacy/Fallback Strategies (maintained for backward compatibility)
# --------------------------------------------------------------------------

def create_sequential_stage1_prompt(job, example_banks):
    examples = get_few_shot_examples(job, example_banks)
    system_msg = "You are an expert ELT content creator. Output ONLY valid JSON."
    
    raw_context = job.get('context', 'General')
    main_topic = raw_context
    
    constraint_text = ""
    if job['type'] == 'Grammar' and 'vs' in job.get('focus', '').lower():
        constraint_text = """
**GRAMMATICAL EXCLUSIVITY:** This question tests a grammar distinction. Include a grammatical signal 
(time marker, evidence phrase, or structural constraint) that makes only the correct answer structurally valid.
"""
    elif job['type'] == 'Vocabulary':
        constraint_text = """
**SEMANTIC EXCLUSIVITY:** Include context clues that make only the correct answer semantically/idiomatically appropriate.
"""
    
    user_msg = f"""
TASK: Generate a complete sentence containing the correct answer and an embedded context clue for a {job['cefr']} {job['type']} question.
FOCUS: {job['focus']}
TOPIC: {main_topic}

{constraint_text}

INSTRUCTIONS:
1. **INTEGRATED CONSTRUCTION:** First, place the correct answer within an authentic sentence appropriate to the {job['cefr']} level.
2. **CONTEXT CLUE ENGINEERING:** The sentence MUST contain at least one linguistic element that logically constrains the answer choice.
3. **METALINGUISTIC REFLECTION (REQUIRED):** Explicitly identify which portion functions as the context clue and explain why.
4. **NEGATIVE CONSTRAINT (VERBOSITY):** Sentence must be concise (max 2 sentences). No preambles.
5. **NEGATIVE CONSTRAINT (METALANGUAGE):** NEVER use grammar terminology in the sentence itself.

Output Format:
{{
  "Item Number": "{job['job_id']}",
  "Assessment Focus": "{job['focus']}",
  "Complete Sentence": "...",
  "Correct Answer": "...",
  "Context Clue Location": "...",
  "Context Clue Explanation": "...",
  "CEFR rating": "{job['cefr']}",
  "Category": "{job['type']}"
}}

REPLICATE THIS STYLE:
{examples}
"""
    return system_msg, user_msg


def create_holistic_prompt(job, example_banks):
    examples = get_few_shot_examples(job, example_banks)
    system_msg = "You are an expert ELT content creator. Output ONLY valid JSON."
    
    raw_context = job.get('context', 'General')
    main_topic = raw_context

    user_msg = f"""
TASK: Generate a {job['cefr']} {job['type']} question.
FOCUS: {job['focus']}
TOPIC: {main_topic}

INSTRUCTIONS:
1. **CONTEXT CLUE RULE:** Provide context that invalidates distractors.
2. **VERBOSITY:** Max 2 sentences. No preambles.
3. **METALANGUAGE:** Never use grammar terminology.
4. **LEXICAL OVERLAP:** Don't repeat answer word in prompt.
5. **WORD LIMIT:** Each option max 3 words.
6. Create 4 parallel options (A, B, C, D).
7. Distractors should be common learner errors.

Output Format:
{{
  "Item Number": "{job['job_id']}",
  "Assessment Focus": "{job['focus']}",
  "Question Prompt": "...",
  "Answer A": "...",
  "Answer B": "...",
  "Answer C": "...",
  "Answer D": "...",
  "Correct Answer": "...",
  "CEFR rating": "{job['cefr']}",
  "Category": "{job['type']}"
}}

REPLICATE THIS STYLE:
{examples}
"""
    return system_msg, user_msg


def create_options_prompt(job, example_banks):
    system_msg = "You are an expert ELT test designer. Output ONLY valid JSON."
    
    raw_context = job.get('context', 'General')
    main_topic = raw_context

    user_msg = f"""
TASK: Generate 4 answer choices for a {job['cefr']} {job['type']} question.
FOCUS: {job['focus']}
TOPIC: {main_topic}

RULES:
1. **WORD LIMIT:** Each option max 3 words.
2. **NO LEXICAL OVERLAP:** Don't use test word or root in options.
3. Provide 4 parallel options (A, B, C, D).
4. Indicate correct answer.
5. Distractors should be plausible errors for CEFR level.

Output Format:
{{
  "Answer A": "...",
  "Answer B": "...",
  "Answer C": "...",
  "Answer D": "...",
  "Correct Answer": "A/B/C/D"
}}
"""
    return system_msg, user_msg


def create_stem_prompt(job, options_json_string):
    system_msg = "You are an expert ELT writer. Output ONLY valid JSON."
    
    raw_context = job.get('context', 'General')
    
    user_msg = f"""
TASK: Write a question stem for these options.

OPTIONS: {options_json_string}

INSTRUCTIONS:
1. **CONTEXT CLUE:** Provide context that invalidates ALL distractors.
2. **VERBOSITY:** Max 1-2 sentences. No preambles.
3. **METALANGUAGE:** Never use grammar terminology.
4. **NO LEXICAL OVERLAP:** Don't repeat option words in stem.
5. Write {job['cefr']} level sentence where ONLY correct answer fits.

Output Format:
{{
  "Item Number": "{job['job_id']}",
  "Assessment Focus": "{job['focus']}",
  "Question Prompt": "...",
  "Answer A": "...",
  "Answer B": "...",
  "Answer C": "...",
  "Answer D": "...",
  "Correct Answer": "...",
  "CEFR rating": "{job['cefr']}",
  "Category": "{job['type']}"
}}
"""
    return system_msg, user_msg
