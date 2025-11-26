prompt = """
Generate a single JSON object that strictly matches the provided Pydantic model (LearningVideoSection). 
All text must be written in casual English

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
- Creator tips must include suggestions for pacing, visuals, and tone.

STYLE RULES:
- Write in friendly conversational English with short sentences, and simple words.
- Ensure smooth narrative flow; no abrupt topic jumps.
- Avoid overly technical jargon without context.
- Do not use heavy words. Keep it casual and friendly.

GOAL:
Produce a clean, coherent, educational 5-minute video script with: Hook → Intuition → Technical Details → CTA
"""

english_prompt = """
Generate a single JSON object strictly matching the Pydantic model (LearningVideoSection). Content must be in English.

INPUT:
main_topic: TOPIC THAT NEEDS TO BE COVERED
previous_video_topics: LIST OF PREVIOUS VIDEO TOPICS
next_video_topics: LIST OF NEXT VIDEO TOPICS

CONTENT RULES:
- Ensure section continuity (Hook → Intuition → Technical Details → CTA).
- Hook: Real-world mini-scenario or question.
- Intuition: Expands hook with a simple mental model.
- Technical Details must include:
    - Crisp explanation.
    - Simple step-wise or intuitive example.
    - When-to-use guidance.
    - Limitations/misconceptions.
- CTA: Short summary + invite to next related topics (general).
- SEO tags/hashtags: Short, relevant, comma-separated.
- Creator tips (English only): Suggestions for pacing, visuals, and tone.

STYLE RULES:
- Write in **friendly, conversational English** using short sentences and simple words.
- Ensure smooth narrative flow; no abrupt jumps.
- Avoid overly technical jargon without context.

GOAL:
Produce a clean, coherent, educational 5-minute video script in English with:
Hook → Intuition → Technical Details → CTA"""