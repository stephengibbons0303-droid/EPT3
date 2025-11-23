import json
import re

def parse_response(raw_response):
    """
    Takes the raw string from the LLM and converts it into a Python dictionary.
    It aggressively cleans the string to handle Markdown code blocks.
    """
    if not raw_response:
        return None, "Empty response from LLM."

    if raw_response.startswith("Error:"):
        return None, raw_response

    try:
        clean_text = raw_response.strip()
        
        if "```" in clean_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            match = re.search(pattern, clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1).strip()
        
        data = json.loads(clean_text)
        return data, None
        
    except json.JSONDecodeError:
        print(f"FAILED JSON: {raw_response}") 
        return None, "Failed to parse JSON. The AI response was malformed."
    except Exception as e:
        return None, f"Unexpected error parsing output: {str(e)}"


def extract_array_from_response(data):
    """
    Extracts an array from LLM response that might be wrapped in various formats.
    Handles cases like: {"questions": [...]} or {"results": [...]} or just [...]
    Also handles single dict objects that should be wrapped in an array.
    Returns (array, error_message)
    
    Updated to include 'candidates' and 'validated' wrapper keys for new three-stage architecture.
    """
    if data is None:
        return None, "No data to extract from."

    # Already an array
    if isinstance(data, list):
        return data, None

    # Wrapped in a dict with a single key
    if isinstance(data, dict):
        # Try common wrapper keys (including new architecture keys: candidates, validated)
        for key in ['questions', 'candidates', 'validated', 'results', 'items', 'data', 'output', 'batch', 'responses']:
            if key in data and isinstance(data[key], list):
                return data[key], None

        # If dict has only one key and it's a list, extract it
        if len(data) == 1:
            only_value = list(data.values())[0]
            if isinstance(only_value, list):
                return only_value, None

        # Check if this looks like a single item that should be in an array
        # Stage 1 items have: "Item Number", "Assessment Focus", etc.
        # Stage 2 items have: "Item Number", "Candidate A", etc.
        # Stage 3 items have: "Item Number", "Selected Distractor A", etc.
        stage1_fields = ["Item Number", "Assessment Focus", "Complete Sentence", "Correct Answer"]
        stage2_fields = ["Item Number", "Candidate A", "Candidate B", "Candidate C"]
        stage3_fields = ["Item Number", "Selected Distractor A", "Selected Distractor B", "Selected Distractor C"]

        # Check if this dict contains fields from any stage
        has_stage1 = any(field in data for field in stage1_fields)
        has_stage2 = any(field in data for field in stage2_fields)
        has_stage3 = any(field in data for field in stage3_fields)

        if has_stage1 or has_stage2 or has_stage3:
            # This is a single item that should be wrapped in an array
            return [data], None

        # Otherwise return error with what we got
        return None, f"Response is a dict but doesn't contain an array. Keys: {list(data.keys())}"

    return None, f"Response is neither array nor dict. Type: {type(data)}"
