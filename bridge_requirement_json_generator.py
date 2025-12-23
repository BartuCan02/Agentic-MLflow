import json
import openai
import os
import re
import ast


def create_bridge_requirement_json(query):
    """
    Uses OpenAI API to generate a structured JSON for bridge requirements based on the query.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
    Given the following user query about a bridge, generate a JSON object with the following fields:
    - task: (segmentation, classification, etc.)
    - class: (type of bridge, if mentioned)
    - main_segments: (list of main parts/segments)
    - width_of_deck: (if mentioned)
    - height_of_deck: (if mentioned)
    - additional_properties: (dictionary of any other relevant attributes)
    Query: {query}
    JSON:
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2
    )
    # Extract JSON from response
    text = response['choices'][0]['message']['content']
    # Try to extract JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        json_str = match.group(0)
        try:
            result = json.loads(json_str)
        except Exception:
            result = ast.literal_eval(json_str)
        return result
    else:
        return {"error": "No JSON found in response", "raw": text}

if __name__ == "__main__":
    query = input("Enter your bridge-related query: ")
    requirement_json = create_bridge_requirement_json(query)
    print(json.dumps(requirement_json, indent=2))
    # Optionally save to file
    with open("bridge_requirement.json", "w") as f:
        json.dump(requirement_json, f, indent=2)
