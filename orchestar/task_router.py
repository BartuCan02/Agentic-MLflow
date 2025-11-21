from utils.llm_utils import ask_llm


def route_task(user_query: str):
    """
    Asks an LLM (or local logic) to determine if the query
    corresponds to a classification or segmentation task.
    """
    system_prompt = (
        "You are an ML pipeline router. Decide whether the user's query "
        "is about classification (e.g., identifying what an object is) "
        "or segmentation (e.g., isolating or marking parts of an object or image or for instance type of/kind of an object). "
        "Respond with only 'classification' or 'segmentation'."
    )

    decision = ask_llm(system_prompt, user_query)

    # Fallback logic if the LLM output is unclear
    decision = decision.strip().lower()
    if "segment" in user_query.lower() or "separate" in user_query.lower():
        return "segmentation"
    if "what is" in user_query.lower() or "is it" in user_query.lower():
        return "classification"

    return "classification" if "class" in decision else "segmentation" if "segment" in decision else "unknown"
