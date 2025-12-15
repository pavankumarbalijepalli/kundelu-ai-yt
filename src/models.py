from pydantic import BaseModel, Field
from typing import List, Literal, Dict

class ContentWriterResponse(BaseModel):
    title: str = Field(
        ..., 
        description="Video title")
    hook: str = Field(
        ..., 
        description="Hook section: Start with real-world scenario or question to grab attention")
    intuition: str = Field(
        ..., 
        description="High-level mental model or analogy explaining the concept which follows the hook example"
    )
    technical_details: str = Field(
        ..., 
        description=(
            "Unified section containing: explanation, step-wise example, "
            "when to use, and limitations/misconceptions"
            "must follow intiution section"
        )
    )
    cta: str = Field(
        ..., 
        description="Wrap-up, summary, and call to action + pointer to next topics"
    )
    seo_tags: str = Field(
        ...,
        description="5–7 relevant SEO keywords for the YouTube video."
    )
    hashtags: str = Field(
        ...,
        description="5–7 appropriate hashtags for the YouTube video."
    )
    walkthrough_code: str = Field(
        ...,
        description="A python file that contains code snippets separated by comments for walkthrough purposes. Mandatory!"
    )