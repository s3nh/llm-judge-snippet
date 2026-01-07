from deepeval import evaluate
from deepeval. models import GeminiVertexAI
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

# Initialize Gemini 2.5 Flash as the judge
gemini_judge = GeminiVertexAI(
    model="gemini-2.5-flash",
    project="your-gcp-project",
    location="us-central1"
)

# Define evaluation metrics using Gemini as judge
answer_relevancy = AnswerRelevancyMetric(
    model=gemini_judge,
    threshold=0.7  # Minimum acceptable score
)

hallucination = HallucinationMetric(
    model=gemini_judge,
    threshold=0.5
)

toxicity = ToxicityMetric(
    model=gemini_judge,
    threshold=0.1  # Lower is better for toxicity
)

# Create test cases
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris, a beautiful city known for the Eiffel Tower.",
    context=["France is a country in Western Europe.  Its capital is Paris."]
)

# Run evaluation with multiple metrics
results = evaluate(
    test_cases=[test_case],
    metrics=[answer_relevancy, hallucination, toxicity]
)

# Print results
for result in results:
    print(f"Test:  {result.input}")
    print(f"Answer Relevancy Score: {result.metrics['answer_relevancy']. score}")
    print(f"Hallucination Score: {result.metrics['hallucination'].score}")
    print(f"Toxicity Score: {result.metrics['toxicity'].score}")
