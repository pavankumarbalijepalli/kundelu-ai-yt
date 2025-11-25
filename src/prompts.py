prompt = """
Generate a single JSON object that strictly matches the provided Pydantic model (LearningVideoSection). 
All text must be written in casual Telugu using English (Roman) script (e.g., “aithe”, “chuddam”, “inka”, “mari”).

INPUT:
main_topic: TOPIC THAT NEEDS TO BE COVERED
previous_video_topics: LIST OF PREVIOUS VIDEO TOPICS COVERED FOR CONTEXT ON WHERE TO START FROM
next_video_topics: LIST OF NEXT VIDEO TOPICS TO TEASE FOR CONTEXT ON WHERE TO END

CONTENT RULES:
- Maintain continuity between sections: 
  • The hook must introduce a real-world mini-scenario or relatable question.
  • The intuition must directly follow from the hook’s scenario, expanding it with a simple mental model.
  • The technical_details must logically continue from the intuition and include:
       - crisp explanation
       - simple example (step-wise or intuitive)
       - when-to-use guidance
       - limitations or misconceptions
- The CTA should give a short summary + invite learning of the next related topics (general, no list required).
- SEO tags and hashtags should be short, relevant, and comma-separated.
- Creator tips must include suggestions for pacing, visuals, and tone (in English).

STYLE RULES:
- Write in friendly conversational Telugu (Roman script), short sentences, simple words.
- Ensure smooth narrative flow; no abrupt topic jumps.
- Avoid overly technical jargon without context.
- DO NOT use TELUGU script; use only English (Roman) script.
- Do not use heavy telugu words. Keep it casual and friendly.

GOAL:
Produce a clean, coherent, educational 5-minute video script in telugu with:
Hook → Intuition → Technical Details → CTA
"""