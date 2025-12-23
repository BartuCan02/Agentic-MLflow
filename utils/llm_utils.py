"""
utils/llm_utils.py
-------------------
Utility for querying the OpenAI API safely and consistently.

- Works with OpenAI package ≥1.0.0
- Handles missing/invalid API keys gracefully
- Returns plain text content for easy downstream use
"""

import os
import openai
from packaging import version

def ask_llm(system_prompt: str, user_query: str) -> str:
    """Send a prompt to the OpenAI API and return the response text."""

    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment."

    try:
        # Detect which API version style is installed
        openai_version = version.parse(openai.__version__)
        if openai_version >= version.parse("1.0.0"):
            # ✅ New client-style API (openai>=1.0.0)
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        else:
            # ✅ Legacy API (openai<1.0.0)
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"].strip()

    except openai.error.AuthenticationError:
        return "Error: Authentication failed. Please check your OPENAI_API_KEY."
    except openai.error.APIConnectionError:
        return "Error: Could not connect to OpenAI API."
    except openai.error.RateLimitError:
        return "Error: Rate limit exceeded."
    except Exception as e:
        return f"Unexpected error: {str(e)}"
