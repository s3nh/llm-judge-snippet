import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
vertexai.init(project="your-gcp-project", location="us-central1")

# Load Gemini 2.5 Flash as the judge model
judge_model = GenerativeModel("gemini-2.5-flash")

def evaluate_output(original_prompt:  str, model_output: str, criteria: str) -> dict:
    """
    Use Gemini 2.5 Flash as a judge to evaluate an LLM output. 
    
    Args:
        original_prompt: The original prompt given to the model
        model_output: The output to be evaluated
        criteria:  Evaluation criteria (e. g., "accuracy", "helpfulness", "safety")
    
    Returns:
        Dictionary with score and explanation
    """
    
    judge_prompt = f"""You are an expert evaluator.  Your task is to judge the quality of an AI assistant's response. 

## Original Prompt:
{original_prompt}

## AI Response to Evaluate:
{model_output}

## Evaluation Criteria: 
{criteria}

Please evaluate the response and provide: 
1. A score from 1-5 (1=poor, 5=excellent)
2. A detailed explanation for your score
3. Specific suggestions for improvement (if any)

Format your response as: 
SCORE: [number]
EXPLANATION: [your detailed explanation]
IMPROVEMENTS: [suggestions or "None needed"]
"""
    
    response = judge_model. generate_content(judge_prompt)
    return parse_judge_response(response. text)

def parse_judge_response(response_text: str) -> dict:
    """Parse the structured judge response."""
    lines = response_text. strip().split('\n')
    result = {"score": None, "explanation": "", "improvements": ""}
    
    current_field = None
    for line in lines: 
        if line.startswith("SCORE:"):
            result["score"] = int(line.replace("SCORE:", "").strip())
        elif line.startswith("EXPLANATION:"):
            current_field = "explanation"
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("IMPROVEMENTS:"):
            current_field = "improvements"
            result["improvements"] = line.replace("IMPROVEMENTS:", "").strip()
        elif current_field: 
            result[current_field] += " " + line.strip()
    
    return result

# Example usage
if __name__ == "__main__":
    # The original prompt given to some LLM
    prompt = "Explain quantum computing in simple terms."
    
    # The output from the LLM being evaluated
    output_to_judge = """Quantum computing uses quantum bits or qubits that can exist 
    in multiple states simultaneously, unlike classical bits which are either 0 or 1. 
    This allows quantum computers to process many possibilities at once."""
    
    # Evaluation criteria
    criteria = """
    - Accuracy: Is the information factually correct?
    - Clarity: Is the explanation easy to understand?
    - Completeness: Does it cover the key concepts?
    """
    
    result = evaluate_output(prompt, output_to_judge, criteria)
    print(f"Score: {result['score']}/5")
    print(f"Explanation: {result['explanation']}")
    print(f"Improvements: {result['improvements']}")
