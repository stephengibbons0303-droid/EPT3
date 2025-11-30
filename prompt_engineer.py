import json
import random
import pandas as pd
import re


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

# =============================================================================
# THREE NEW FUNCTIONS TO ADD TO prompt_engineer.py
# =============================================================================
# 
# INSTRUCTIONS:
# 1. Open your existing prompt_engineer.py file
# 2. Locate the get_few_shot_examples() function (around line 15-50)
# 3. Copy all three functions below
# 4. Paste them immediately AFTER get_few_shot_examples() ends
# 5. Your existing functions remain unchanged
#
# =============================================================================


def create_vocab_list_stage1_prompt(job_list, question_form):
    """
    STAGE ONE for vocabulary list upload: Generates complete sentences with correct answers.
    Includes question form constraints based on user selection.
    """
    question_form_instructions = {
        "Random Mix": """
QUESTION FORM VARIETY: Use diverse question forms across the batch. Include:
- Simple gap fill (most common): "Our new sidewalk is made of ___________."
- Definition through function: "The fuselage is where the ___________."
- Cause-Effect completion: "The wing is damaged. The ___________ can't leave on time."
- Dialogue completion: "Sue: Did everything go alright? Mary: It was ___________, everyone enjoyed themselves."
- Logical relationship: "Tom is not ___________. He has an abundant amount of money."
""",
        "Simple gap fill": """
QUESTION FORM CONSTRAINT: ALL questions must use simple gap fill format.
Example: "Our new sidewalk is made of ___________."
The sentence should provide clear context that makes the target word the logical choice.
""",
        "Definition through function/description": """
QUESTION FORM CONSTRAINT: ALL questions must use definition through function/description format.
Example: "The fuselage is where the ___________."
The target vocabulary word should complete the functional definition.
""",
        "Cause-Effect completion": """
QUESTION FORM CONSTRAINT: ALL questions must use cause-effect completion format.
Example: "The wing is damaged. The ___________ can't leave on time."
The target word should logically complete the consequence.
""",
        "Dialogue completion": """
QUESTION FORM CONSTRAINT: ALL questions must use dialogue completion format.
Example: "Sue: Did everything go alright? Mary: It was ___________, everyone enjoyed themselves."
The target word should fit naturally in conversational context.
""",
        "Logical relationship completion": """
QUESTION FORM CONSTRAINT: ALL questions must use logical relationship completion format.
Example: "Tom is not ___________. He has an abundant amount of money."
The target word should complete the logical connection between clauses.
"""
    }
    
    form_instruction = question_form_instructions.get(question_form, question_form_instructions["Random Mix"])
    
    system_msg = f"""You are an expert ELT content creator. You will generate exactly {len(job_list)} complete test questions in a single JSON response targeting specific vocabulary items provided by the user.

CRITICAL: Your entire response must be a JSON object with a "questions" key containing an array of exactly {len(job_list)} question objects."""
    
    job_specs = []
    for job in job_list:
        job_specs.append({
            "job_id": job['job_id'],
            "cefr": job['cefr'],
            "target_vocabulary": job['target_vocabulary'],
            "definition": job.get('definition', ''),
            "part_of_speech": job.get('part_of_speech', '')
        })
    
    user_msg = f"""
TASK: Create exactly {len(job_list)} vocabulary test questions targeting specific vocabulary items.

VOCABULARY TARGETS (one question for each):
{json.dumps(job_specs, indent=2)}

{form_instruction}

GENERATION INSTRUCTIONS FOR EACH QUESTION:

1. **TARGET VOCABULARY INTEGRATION:** The "Complete Sentence" must contain the target vocabulary item in a natural, authentic context appropriate for the CEFR level. Use the provided definition to ensure accurate usage.

2. **PART OF SPEECH MATCHING:** Ensure the target vocabulary is used in the correct grammatical form matching the "Part of Speech" field.

3. **SEMANTIC CONTEXT CLUES:** Include context clues that make only the target vocabulary semantically appropriate while keeping all options grammatically valid.

4. **DIFFICULTY CALIBRATION:** Match sentence complexity and vocabulary sophistication to the CEFR level.

5. **NEGATIVE CONSTRAINT (VERBOSITY):** Sentences must be concise (max 2 sentences for simple forms, max 3 for dialogue). No preambles.

6. **NEGATIVE CONSTRAINT (DEFINITION LEAK):** Do NOT include the definition text directly in the sentence.

7. **ANTI-REPETITION:** Each question must use a unique scenario and context.

MANDATORY OUTPUT FORMAT:
{{
  "questions": [
    {{
      "Item Number": "...",
      "Target Vocabulary": "...",
      "Complete Sentence": "...",
      "Correct Answer": "...",
      "Context Clue Location": "...",
      "Context Clue Explanation": "...",
      "CEFR rating": "...",
      "Category": "Vocabulary"
    }},
    ... (continue until you have exactly {len(job_list)} question objects)
  ]
}}

VERIFICATION: Count your question objects before submitting. You must have exactly {len(job_list)} items in the "questions" array.
"""
    return system_msg, user_msg
# =============================================================================
# HELPER FUNCTIONS FOR DUAL SELECTION
# =============================================================================

def get_first_word(vocab_item):
    """
    Extract the first word from multi-word vocabulary items.
    Examples: "clear up" → "clear", "belong (to)" → "belong", "add on" → "add"
    """
    # Remove parentheses and their contents
    cleaned = re.sub(r'\([^)]*\)', '', vocab_item)
    # Split and get first word
    words = cleaned.strip().split()
    return words[0] if words else vocab_item


def get_initial_letter(vocab_item):
    """
    Get the first letter of the first word.
    """
    first_word = get_first_word(vocab_item)
    return first_word[0].lower() if first_word else ''


def get_phonetic_similar_letters(letter):
    """
    Return phonetically similar letters for fallback matching.
    Used when initial letter pool is limited.
    """
    phonetic_groups = {
        'c': ['k', 'q'],  # hard c sound
        'k': ['c', 'q'],
        'q': ['c', 'k'],
        's': ['c', 'z'],  # soft s/c sound
        'z': ['s'],
        'f': ['ph'],
        'ph': ['f'],
        'j': ['g'],  # soft g sound
        'g': ['j'],
        'i': ['y'],  # initial vowel sounds
        'y': ['i']
    }
    return phonetic_groups.get(letter.lower(), [])


def python_select_by_pos(vocab_df, target_vocab, target_pos, max_items=4):
    """
    Select distractors by matching part of speech.
    Returns up to max_items vocabulary items.
    """
    target_vocab_lower = target_vocab.lower().strip()
    target_pos_lower = target_pos.lower().strip()
    
    # Filter by same part of speech
    same_pos = vocab_df[
        vocab_df['Part of Speech'].str.lower().str.strip() == target_pos_lower
    ]
    
    # Exclude the target vocabulary
    same_pos = same_pos[
        same_pos['Base Vocabulary Item'].str.lower().str.strip() != target_vocab_lower
    ]
    
    # Randomly select up to max_items
    if len(same_pos) >= max_items:
        selected = same_pos.sample(n=max_items)
    else:
        selected = same_pos
    
    return selected['Base Vocabulary Item'].tolist()


def python_select_by_initial_letter(vocab_df, target_vocab, max_items=4, exclude_items=None):
    """
    Select distractors by matching initial letter of first word.
    Includes phonetic fallback if pool is limited.
    
    Args:
        vocab_df: The vocabulary dataframe
        target_vocab: The target vocabulary item
        max_items: Maximum number to select
        exclude_items: List of items to exclude (already selected by POS)
    
    Returns:
        List of vocabulary items matching initial letter
    """
    if exclude_items is None:
        exclude_items = []
    
    target_vocab_lower = target_vocab.lower().strip()
    target_letter = get_initial_letter(target_vocab)
    
    # Get all vocab items with same initial letter
    def matches_initial_letter(vocab_item):
        return get_initial_letter(vocab_item) == target_letter
    
    same_letter = vocab_df[
        vocab_df['Base Vocabulary Item'].apply(matches_initial_letter)
    ]
    
    # Exclude target vocab and already-selected items
    exclude_lower = [item.lower().strip() for item in exclude_items + [target_vocab]]
    same_letter = same_letter[
        ~same_letter['Base Vocabulary Item'].str.lower().str.strip().isin(exclude_lower)
    ]
    
    # If we have enough, select and return
    if len(same_letter) >= max_items:
        selected = same_letter.sample(n=max_items)
        return selected['Base Vocabulary Item'].tolist()
    
    # If pool is limited, collect what we have
    candidates = same_letter['Base Vocabulary Item'].tolist()
    
    # PHONETIC FALLBACK: Try phonetically similar letters
    phonetic_letters = get_phonetic_similar_letters(target_letter)
    
    for phon_letter in phonetic_letters:
        if len(candidates) >= max_items:
            break
        
        def matches_phonetic_letter(vocab_item):
            return get_initial_letter(vocab_item) == phon_letter
        
        phonetic_matches = vocab_df[
            vocab_df['Base Vocabulary Item'].apply(matches_phonetic_letter)
        ]
        
        phonetic_matches = phonetic_matches[
            ~phonetic_matches['Base Vocabulary Item'].str.lower().str.strip().isin(
                exclude_lower + [c.lower() for c in candidates]
            )
        ]
        
        needed = max_items - len(candidates)
        if len(phonetic_matches) > 0:
            additional = phonetic_matches.sample(n=min(needed, len(phonetic_matches)))
            candidates.extend(additional['Base Vocabulary Item'].tolist())
    
    return candidates[:max_items]

# =============================================================================
# MAIN STAGE 2 FUNCTION - DUAL SELECTION + LLM SUPPLEMENTATION
# =============================================================================

def create_vocab_list_stage2_prompt(job_list, stage1_outputs, vocabulary_list_df):
    """
    STAGE TWO for vocabulary list: DUAL SELECTION STRATEGY
    
    Python performs two selection runs:
    1. Select up to 4 items by matching Part of Speech
    2. Select up to 4 items by matching initial letter (with phonetic fallback)
    
    LLM supplements to reach 8 total candidates:
    - Prioritize antonyms of target vocabulary (for adj/verb/adverb/prep)
    - Fill remainder with synonyms of Python-selected distractors
    """
    system_msg = f"""You are an expert ELT test designer specializing in vocabulary assessment. You will supplement pre-selected vocabulary candidates with additional distractors for exactly {len(job_list)} questions.

CRITICAL: Your entire response must be a JSON object with a "candidates" key containing an array of exactly {len(job_list)} candidate sets."""
    
    # PYTHON DUAL SELECTION: POS + Initial Letter
    pre_selected_data = []
    
    for i, job in enumerate(job_list):
        stage1_data = stage1_outputs[i]
        target_vocab = job['target_vocabulary']
        target_pos = job['part_of_speech']
        
        # SELECTION RUN 1: By Part of Speech (max 4)
        pos_selected = python_select_by_pos(
            vocabulary_list_df,
            target_vocab,
            target_pos,
            max_items=4
        )
        
        # SELECTION RUN 2: By Initial Letter (max 4, exclude POS-selected)
        letter_selected = python_select_by_initial_letter(
            vocabulary_list_df,
            target_vocab,
            max_items=4,
            exclude_items=pos_selected
        )
        
        total_python = len(pos_selected) + len(letter_selected)
        needed_from_llm = max(0, 8 - total_python)
        
        pre_selected_data.append({
            "Item Number": stage1_data.get("Item Number"),
            "Target Vocabulary": target_vocab,
            "Part of Speech": target_pos,
            "Complete Sentence": stage1_data.get("Complete Sentence"),
            "Correct Answer": stage1_data.get("Correct Answer"),
            "POS-selected": pos_selected,
            "Letter-selected": letter_selected,
            "Total from vocab list": total_python,
            "Needed from LLM": needed_from_llm
        })
    
    # LLM SUPPLEMENTATION TASK
    user_msg = f"""
TASK: Supplement the pre-selected vocabulary candidates to create a pool of exactly 8 distractor candidates for each question.

INPUT (Complete sentences with pre-selected vocabulary):
{json.dumps(pre_selected_data, indent=2)}

BACKGROUND:

Students have recently studied this vocabulary list and may have memorized it. They are MORE likely to:
- Confuse meanings of familiar words from the list
- Spot unfamiliar words (not in the list) as obvious wrong answers

Therefore, the Python code has already selected distractors FROM the vocabulary list using two strategies:
1. **POS-selected**: Items with the same part of speech as the target
2. **Letter-selected**: Items with similar initial letters (phonetic confusion for lower-level learners)

YOUR TASK:

For EACH question, you must supplement the pre-selected candidates to reach exactly 8 total candidates.

The number you need to add is shown in "Needed from LLM" for each question.

SUPPLEMENTATION STRATEGY:

**Priority 1: Generate antonyms of the target vocabulary** (when applicable)
- For adjectives, verbs, adverbs, and prepositions, try to create at least ONE antonym
- Example: target "calm" → generate "agitated" or "turbulent"
- Example: target "dangerous" → generate "safe"
- Antonyms should match the exact grammatical form of the target (same tense, number, etc.)

**Priority 2: Generate synonyms/near-synonyms of the Python-selected distractors**
- Look at the items in "POS-selected" and "Letter-selected"
- Generate words semantically similar to THOSE items (not the target)
- Example: If "climate" is in POS-selected, generate "weather" or "temperature"
- This maintains the "familiar vocabulary confusion" strategy

CRITICAL CONSTRAINTS:

1. **EXACT COUNT**: You must add EXACTLY the number shown in "Needed from LLM"

2. **WORD COUNT LIMIT**: Each generated candidate must be MAXIMUM 1 WORD (no slashes, no multiple forms)

3. **EXACT INFLECTIONAL FORM MATCHING (CRITICAL)**: 
   - Generated candidates must match the PRECISE grammatical form of the target vocabulary
   - Same tense, aspect, person, number, and voice
   - Examples:
     * Target: "blew" (past) → Generate: "boiled", "fried", "baked" (NOT "boil", "fry", "bake")
     * Target: "blowing" (gerund) → Generate: "boiling", "frying", "baking" (NOT "boil", "fry", "bake")
     * Target: "blows" (3rd singular) → Generate: "boils", "fries", "bakes" (NOT "boil", "fry", "bake")
     * Target: "blow" (base) → Generate: "boil", "fry", "bake" (NOT "boils", "boiled", "boiling")
   
   **FORM IDENTIFICATION RULE**: 
   - Check the Complete Sentence context
   - If target follows "to" or modal (can/will/should) → BASE FORM
   - If target follows "he/she/it" → 3RD SINGULAR (-s/-es)
   - If target has time marker "yesterday" → PAST TENSE (-ed or irregular)
   - If target follows "is/was/been" → GERUND (-ing) or PAST PARTICIPLE (-ed/-en)

4. **SINGLE WORD ONLY**: 
   - NEVER use slashes: "blow/blew/blown" is WRONG
   - NEVER list multiple forms: "boil or boiled" is WRONG
   - Each candidate must be ONE inflected form: "boiled" is CORRECT

5. **NO NEAR-SYNONYMS OF TARGET**: Do NOT generate near-synonyms or direct synonyms of the target vocabulary (except for intentional antonyms)

6. **PRESERVE PRE-SELECTED**: Include ALL items from both "POS-selected" and "Letter-selected" in your output, then add your generated items

FORM MATCHING EXAMPLES:

Example 1 - Past Tense:
- Complete Sentence: "The wind **blew** all day."
- Target: "blew" (past tense of "blow")
- POS-selected: ["calm", "clear", "humid"] (adjectives - WRONG, reject these)
- Letter-selected: ["beach", "bake"]
- LLM must generate: "boiled", "fried" (past tense verbs)
- WRONG: "boil", "fry" (base form)
- WRONG: "blow/blew/blown" (multiple forms)

Example 2 - Base Form:
- Complete Sentence: "I need to **clean** the kitchen."
- Target: "clean" (base form after "to")
- POS-selected: ["cook", "wash", "dust"]
- LLM can add: "scrub", "tidy" (base form verbs)
- WRONG: "scrubbed", "tidied" (past tense)
- WRONG: "scrubbing", "tidying" (gerund)

Example 3 - 3rd Person Singular:
- Complete Sentence: "She **cooks** every evening."
- Target: "cooks" (3rd person singular)
- Letter-selected: ["climbs", "cleans", "closes"]
- LLM must generate: "bakes", "boils", "fries" (3rd person with -s)
- WRONG: "bake", "boil", "fry" (base form)
- WRONG: "baked", "boiled", "fried" (past tense)

Example 4 - Gerund:
- Complete Sentence: "I enjoy **reading** books."
- Target: "reading" (gerund after "enjoy")
- POS-selected: ["writing", "playing", "singing"]
- LLM can add: "cooking", "dancing" (gerund with -ing)
- WRONG: "cook", "dance" (base form)
- WRONG: "cooked", "danced" (past tense)

MANDATORY OUTPUT FORMAT:
{{
  "candidates": [
    {{
      "Item Number": "...",
      "Candidate A": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Candidate B": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Candidate C": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Candidate D": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Candidate E": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Candidate F": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Candidate G": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Candidate H": "...[from POS-selected or Letter-selected or LLM-generated]...",
      "Source Notes": "e.g., 'A-D from vocab list, E antonym of target, F-H synonyms of list items'"
    }},
    ... (exactly {len(job_list)} candidate sets)
  ]
}}

EXAMPLE:

Input:
- Target: "dangerous" (adjective)
- POS-selected: ["calm", "clear", "humid"]
- Letter-selected: ["deep", "during"]
- Total: 5, Needed: 3

Output Candidates:
- A: "calm" (from POS)
- B: "clear" (from POS)
- C: "humid" (from POS)
- D: "deep" (from Letter)
- E: "during" (from Letter)
- F: "safe" (LLM antonym of "dangerous")
- G: "peaceful" (LLM synonym of "calm")
- H: "transparent" (LLM synonym of "clear")

VERIFICATION: You must generate exactly {len(job_list)} candidate sets with exactly 8 candidates each.
"""
    return system_msg, user_msg


# =============================================================================
# INTEGRATION NOTES
# =============================================================================
# 
# This completely replaces your current create_vocab_list_stage2_prompt function.
# 
# Make sure to add all the helper functions above it in your prompt_engineer.py:
# - get_first_word()
# - get_initial_letter()
# - get_phonetic_similar_letters()
# - python_select_by_pos()
# - python_select_by_initial_letter()
# 
# Then replace the main Stage 2 function with this new version.
# 
# The rest of your Tab 4 code (Stage 1, Stage 3, final assembly) remains unchanged.
# Stage 3 will still receive 8 candidates and select the best 5 for validation,
# then narrow down to the final 3 distractors.
# =============================================================================

def create_vocab_list_stage3_prompt(job_list, stage1_outputs, stage2_outputs):
    """
    STAGE THREE for vocabulary list: Binary validation with PRAGMATIC FILTERING.
    IMPROVED: Explicit collocation and semantic compatibility checks.
    """
    system_msg = f"""You are an expert English vocabulary validator with deep knowledge of collocations and pragmatic appropriateness. You will evaluate candidate distractors for exactly {len(job_list)} vocabulary questions and return your validated selections in a JSON object with a "validated" key."""
    
    validation_input = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        complete_sentence = s1.get("Complete Sentence", "")
        correct_answer = s1.get("Correct Answer", "")
        
        validation_input.append({
            "Item Number": s1.get("Item Number", ""),
            "Target Vocabulary": s1.get("Target Vocabulary", ""),
            "Complete Sentence": complete_sentence,
            "Correct Answer": correct_answer,
            "Candidate A": s2.get("Candidate A", ""),
            "Candidate B": s2.get("Candidate B", ""),
            "Candidate C": s2.get("Candidate C", ""),
            "Candidate D": s2.get("Candidate D", ""),
            "Candidate E": s2.get("Candidate E", ""),
            "Candidate F": s2.get("Candidate F", ""),
            "Candidate G": s2.get("Candidate G", ""),
            "Candidate H": s2.get("Candidate H", ""),
            "CEFR": job['cefr']
        })
    
    user_msg = f"""
TASK: Validate candidate distractors for ALL {len(job_list)} VOCABULARY questions and select the final three distractors per question.

VALIDATION INPUT:
{json.dumps(validation_input, indent=2)}

VALIDATION PROCEDURE:

For EACH question, test ALL EIGHT candidates using this THREE-TIER FILTER:

**TIER 1: GRAMMATICAL CORRECTNESS (Required)**
Replace the Correct Answer with each candidate individually.
- REJECT if grammatically incorrect (wrong part of speech, agreement errors, etc.)
- RETAIN if grammatically correct (proper sentence structure)

**TIER 2: MULTIPLE CORRECT ANSWER CHECK (NEW - Critical)**
For candidates that pass Tier 1, ask: "Would a test examiner accept this as a correct answer?"

REJECT if the candidate is:
- A plausible synonym of the correct answer
- Semantically appropriate for the context
- Would make logical sense in the sentence

RETAIN if the candidate is:
- Grammatically correct BUT semantically inappropriate
- Creates a pragmatically odd/unnatural sentence
- Would NOT be accepted as a correct answer

**CRITICAL DISTINCTION EXAMPLES**:

Example 1: "We spent our day relaxing on the __________" (Correct: beach)

REJECT THESE (too plausible, multiple correct answers):
- "patio" ❌ REJECT - examiner would accept this (plausible location)
- "deck" ❌ REJECT - examiner would accept this (plausible location)
- "balcony" ❌ REJECT - examiner would accept this (plausible location)

RETAIN THESE (grammatically fine, semantically wrong = good distractors):
- "professional" ✓ RETAIN - grammatically correct, semantically odd (people don't relax on professions)
- "climate" ✓ RETAIN - grammatically correct, semantically wrong (can't relax on climate)

REJECT THESE (too obviously wrong):
- "builder" ❌ REJECT - wrong category (person, not place) - too easy to eliminate
- "high school" ❌ REJECT - pragmatically absurd in this context - too easy to eliminate

Example 2: "After the storm, the sea was completely __________" (Correct: calm)

REJECT THESE (too plausible, multiple correct answers):
- "still" ❌ REJECT - examiner would accept this (plausible descriptor)
- "peaceful" ❌ REJECT - examiner would accept this (plausible synonym)
- "quiet" ❌ REJECT - examiner would accept this (plausible descriptor)

RETAIN THESE (grammatically fine, semantically wrong = good distractors):
- "professional" ✓ RETAIN - grammatically correct, semantically inappropriate (sea can't be professional)
- "confusing" ✓ RETAIN - grammatically correct, semantically odd (unusual adjective for sea)

REJECT THESE (too obviously wrong):
- "fish" ❌ REJECT - wrong part of speech, grammatically broken
- "builder" ❌ REJECT - wrong part of speech, obviously wrong

Example 3: "The instructions need to be __________" (Correct: clear)

REJECT THESE (too plausible):
- "simple" ❌ REJECT - examiner would accept
- "detailed" ❌ REJECT - examiner would accept

RETAIN THESE (good distractors):
- "freezing" ✓ RETAIN - grammatically correct adjective, semantically incompatible with instructions
- "own" ✓ RETAIN - grammatically correct, semantically wrong

REJECT THESE (too obvious):
- "fish" ❌ REJECT - wrong part of speech

**TIER 3: PEDAGOGICAL QUALITY (Final Selection)**
From candidates that pass Tiers 1 and 2, select the BEST THREE based on:
- Variety of semantic incompatibility types
- CEFR-appropriate challenge (requires thought to eliminate, not instantly obvious)
- Avoid selecting multiple candidates from same semantic field

MANDATORY OUTPUT FORMAT:
{{
  "validated": [
    {{
      "Item Number": "...",
      "Selected Distractor A": "...",
      "Selected Distractor B": "...",
      "Selected Distractor C": "...",
      "Validation Notes": "Brief explanation: [X] rejected for collocation violations, [Y] rejected for semantic absurdity, selected [ABC] for plausibility"
    }},
    ... (exactly {len(job_list)} validated sets)
  ]
}}

VERIFICATION: You must provide exactly {len(job_list)} validated distractor sets with exactly 3 selected distractors each.
"""
    return system_msg, user_msg

# INTEGRATION INSTRUCTIONS:
# Replace the existing create_vocab_list_stage3_prompt function in prompt_engineer.py
# This version adds explicit pragmatic and collocational filtering
# Expected improvement: 34% → 60-70% plausible distractors

# =============================================================================
# END OF NEW FUNCTIONS
# =============================================================================

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

3. **LEXICAL OVERLAP PROHIBITION:** Do NOT use any words that appear AFTER the blank in the Complete Sentence.

Examples of VIOLATIONS:
- "Do you have ____ sugar?" → WRONG: "a sugar", "the sugar" (repeats "sugar")
- "She doesn't have ____ friends" → WRONG: "no friends", "zero friends" (repeats "friends")
- "He ____ have won" → WRONG: "might have", "should won" (repeats "have" or "won")

VERIFICATION: Before submitting, scan the post-blank text and ensure no candidates repeat those words.

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

2. **WORD FORM ENFORCEMENT:** If the Assessment Focus is "Word Form", candidates MUST include different word classes derived from related roots.

Examples:
- Assessment Focus: "Word Form (noun/verb/adj)" → Correct answer: "complement" (verb)
  → CORRECT candidates: "compliment" (noun), "complementary" (adj), "completion" (noun)
  → WRONG candidates: "criticize", "contrast", "complicate" (all verbs, same class)

- Assessment Focus: "Word Form (noun/verb/adj)" → Correct answer: "seasoned" (adj)
  → CORRECT candidates: "season" (noun), "seasonal" (adj), "seasoning" (noun)
  → WRONG candidates: "baked", "chopped", "froze" (all verbs, same class)

If Assessment Focus is NOT "Word Form", then candidates should be same word class as correct answer.

3. **EXACT INFLECTIONAL FORM MATCHING:** ALL candidates must match the PRECISE grammatical form of the correct answer - same tense, aspect, person, number, and voice. Examples:
   - Correct: "brush" (base) → Candidates: "comb", "wash", "scrub" (NOT "combs", "washes", "scrubs")
   - Correct: "brushes" (3rd singular) → Candidates: "combs", "washes", "scrubs" (NOT "comb", "wash", "scrub")
   - Correct: "plowed through" (past) → Candidates: "broke through", "pushed through" (NOT "break through", "push through")
   **FORM IDENTIFICATION RULE:** Before generating candidates, check: Does the correct answer follow 'to', a modal (can/will/should/want to), or a plural subject (I/you/we/they)? If YES, it is BASE FORM - all candidates must be base form with NO -s/-es/-ed endings.
   

4. **SAME WORD CLASS AND GRAMMATICAL FORM REQUIRED:** All candidates must match the exact grammatical structure of the correct answer. If the correct answer is a present tense third person singular verb, all candidates must be present tense third person singular verbs.

5. **SEMANTIC FIELD PROXIMITY:** Generate alternatives from the SAME semantic field as the correct answer. For example:
   - If the correct answer is "take a look at" (visual observation), candidates should be other observation phrases like "behold", "gaze upon", "pay heed to"
   - If the correct answer is "postpone" (delay action), candidates should be other temporal or planning verbs like "possess", "position", "compose"

6. **CONTEXTUAL INAPPROPRIATENESS TYPES:**
   - **Register Conflict:** Formal/archaic terms in casual contexts (e.g., "behold" for "look at")
   - **Collocational Mismatch:** Verbs that don't combine naturally with following nouns (e.g., "make breakfast" vs "do breakfast")
   - **Idiomatic Violation:** Phrases that violate standard usage patterns (e.g., "pay heed to" with visual objects)

7. **PHONETIC AND SEMANTIC ALTERNATIVES PERMITTED:** For higher-level vocabulary (B1+), include at least one phonetically similar alternative that matches lexical sophistication (e.g., "possess" or "position" as alternatives to "postpone").

8. **NO LEXICAL OVERLAP:** Do not use any form of the correct answer word or its root in candidates.

9. **POST-BLANK LEXICAL OVERLAP PROHIBITION:** Do NOT use any words that appear AFTER the blank in the Complete Sentence.

Examples of VIOLATIONS:
- "I need to buy ____ apples" → WRONG: "some apples", "fresh apples" (repeats "apples")
- "plant a tree for more ____" → WRONG: "more shade", "extra shade" (repeats "more")

VERIFICATION: Scan the post-blank text and ensure no candidates repeat those words.

10. **ANTI-REPETITION:** Avoid using identical candidate words across multiple questions in this batch.

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

4. **UNIQUENESS TEST:** For each candidate, answer this question:
   
   "If a student chose this candidate, could they argue it's grammatically acceptable?"
   - YES = REJECT the candidate (multiple correct answers exist)
   - NO = RETAIN the candidate (clearly grammatically wrong)

Critical examples:
- "It ____ cold outside, so wear a jacket" + candidate "was" → Student argues: "It was cold [earlier], so wear a jacket [now]" → YES, defensible → REJECT
- "It ____ cold outside, so wear a jacket" + candidate "are" → Student argues: "It are cold" → NO, indefensible → RETAIN

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

2. **EXAMINER ACCEPTANCE TEST:** For each reconstructed sentence, answer this question:
   
   "Would a test examiner award full marks if a student chose this candidate?"
   - YES = REJECT the candidate (acceptable alternative answer)
   - NO = RETAIN the candidate (clearly inappropriate)

Critical examples:
- "He felt very ____ after good news" + candidate "excited" → Examiner accepts "excited" as valid positive emotion → YES → REJECT
- "He felt very ____ after good news" + candidate "confused" → Examiner rejects "confused" (contradicts context) → NO → RETAIN
- "plant a tree for more ____" + candidate "color" → Examiner accepts "color" (cherry blossoms provide color) → YES → REJECT

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
