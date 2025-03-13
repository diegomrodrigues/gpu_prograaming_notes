import os

os.environ["MOCK_API"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "pollo-create-topics"


from pollo.agents.draft.writer import generate_drafts_from_topics

drafts = generate_drafts_from_topics(
    directory="01. Data Parallelism"
)


#from pollo.agents.topics.generator import generate_topics_from_pdfs

#topics = generate_topics_from_pdfs(
#    directory="01. Data Parallelism",
#    perspectives=[
#        "Code",
#        "Math"
#    ],
#    json_per_perspective=1
#)