from langchain_google_genai import ChatGoogleGenerativeAI
from models import LearningVideoSection
from email_handler import send_email
from datetime import timedelta as td
from datetime import datetime as dt
from utils import log, content_map
from prompts import prompt
import markdown
import json
import os

os.chdir('..')

from dotenv import load_dotenv
status = load_dotenv('.env')
if status:
    log("Environment variables loaded.")
else:
    log("Failed to load environment variables.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
log("LLM initialized.")
llm = llm.with_structured_output(LearningVideoSection)
log("LLM wrapped with structured output.")

today_topics = content_map.get(dt.now().strftime('%Y-%m-%d'), None)
yesterday_topics = content_map.get((dt.now() - td(days=1)).strftime('%Y-%m-%d'), None)
tomorrow_topics = content_map.get((dt.now() + td(days=1)).strftime('%Y-%m-%d'), None)
log(f"Processing topic: {today_topics}")

topic_1 = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": f"main_topic: {today_topics[0]}\nprevious_video_topics: {yesterday_topics}\nnext_video_topics: {tomorrow_topics}"}])
topic_2 = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": f"main_topic: {today_topics[1]}\nprevious_video_topics: {yesterday_topics}\nnext_video_topics: {tomorrow_topics}"}])
log("LLM response received.")

content = ""
_json = json.loads(topic_1.model_dump_json())
for key, value in _json.items():
    if key in ['seo_tags', 'hashtags']:
        content += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
    elif key == 'walkthrough_code':
        continue
    else:
        content += f"## {key.replace('_', ' ').upper()}\n\n{value}\n\n"
content += f"<h2>Walkthrough Code:</h2> \n\n<pre><code>\n{value}\n</code></pre>\n\n"

_json = json.loads(topic_2.model_dump_json())
for key, value in _json.items():
    if key in ['seo_tags', 'hashtags']:
        content += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
    elif key == 'walkthrough_code':
        continue
    else:
        value = '\n'.join(['- '+line for line in value.split('. ')])
        content += f"## {key.replace('_', ' ').upper()}\n\n{value}\n\n"
content += f"<h2>Walkthrough Code:</h2> \n\n<pre><code>\n{value}\n</code></pre>\n\n"

send_email(markdown.markdown(content), ' & '.join([today_topic.split('>')[-1].strip() for today_topic in today_topics]))