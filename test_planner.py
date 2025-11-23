import random

def create_job_list(
    total_questions, 
    q_type, 
    cefr_target, 
    selected_focus_list, 
    context_topic,
    generation_strategy
):
    """
    Generates a list of job dictionaries with unique topics for each job 
    to prevent content repetition across the batch.
    
    Topic variance is the primary anti-repetition mechanism, leveraging the 
    batch processing model's cross-question awareness. Style micro-contexts 
    have been removed to prevent contamination of Assessment Focus labels.
    """
    job_list = []
    
    # Topic Variance (Semantic Domains) - prevents thematic repetition
    random_domains = [
        "Health & Fitness", "Technology & Computers", "Cooking & Food", 
        "Money & Shopping", "Daily Routine", "Art & Music", 
        "Weather & Nature", "Work & Jobs", "Education & Learning", 
        "Transport & Cities", "Family & Relationships", "Current Events"
    ]
    
    # Check if user provided a specific topic
    user_provided_topic = True
    if not context_topic or context_topic.strip() == "":
        user_provided_topic = False
    
    for i in range(total_questions):
        current_focus = random.choice(selected_focus_list)
        job_id = f"{q_type[0].upper()}{cefr_target}-{i+1}"
        
        if user_provided_topic:
            # Use user-specified topic for all questions in batch
            main_topic = context_topic
        else:
            # Cycle through random domains to ensure topic diversity
            current_domain = random_domains[i % len(random_domains)]
            main_topic = current_domain
        
        job = {
            "job_id": job_id,
            "type": q_type,
            "cefr": cefr_target,
            "focus": current_focus,
            "context": main_topic,
            "strategy": generation_strategy
        }
        
        job_list.append(job)
        
    return job_list
