from langchain_google_genai import ChatGoogleGenerativeAI
from models import LearningVideoSection
from email_handler import send_email
from prompts import prompt
from utils import log
import markdown
import pickle
import json
import os

os.chdir('..')

from dotenv import load_dotenv
load_dotenv('.env')
log("Environment variables loaded.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
log("LLM initialized.")
llm = llm.with_structured_output(LearningVideoSection)
log("LLM wrapped with structured output.")

topics = pickle.load(open('data/content_ideas.pkl', 'rb'))
topic = topics['Unfinished'].pop(0)
topics['finished'].append(topic)
pickle.dump(topics, open('data/content_ideas.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
log("Content ideas updated.")

log(f"Processing topic: {topic}")

response = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": topic}])
log("LLM response received.")

content = ""
_json = json.loads(response.model_dump_json())
for key, value in _json.items():
    if key not in ['seo_tags', 'hashtags', 'creator_tips']:
        content += f"## {key.replace('_', ' ').title()}\n\n{value}\n\n"
    else:
        content += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
send_email(markdown.markdown(content), topic.split('>')[-1].strip())