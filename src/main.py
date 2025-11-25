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

today_topic = content_map.get(dt.now().strftime('%Y-%m-%d'), None)
yesterday_topic = content_map.get((dt.now() - td(days=1)).strftime('%Y-%m-%d'), None)
tomorrow_topic = content_map.get((dt.now() + td(days=1)).strftime('%Y-%m-%d'), None)
log(f"Processing topic: {today_topic}")

response = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": f"main_topic: {today_topic}\nprevious_video_topics: {yesterday_topic}\nnext_video_topics: {tomorrow_topic}"}])
log("LLM response received.")

content = ""
_json = json.loads(response.model_dump_json())
for key, value in _json.items():
    if key not in ['seo_tags', 'hashtags', 'creator_tips']:
        content += f"## {key.replace('_', ' ').title()}\n\n{value}\n\n"
    else:
        content += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
send_email(markdown.markdown(content), today_topic.split('>')[-1].strip())