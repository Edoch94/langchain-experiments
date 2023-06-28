import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap

#%%
# Load the HuggingFaceHub API token from the .env file
# --------------------------------------------------------------
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# %%
# Load the text
# --------------------------------------------------------------
from example_text import TEXT_ORIG
text_input=TEXT_ORIG
print(len(text_input.split(' ')))


#%%
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------
repo_id = "tiiuae/falcon-7b"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 1000}
) # type: ignore

#%%
# Load an OpenAI model for comparison
# --------------------------------------------------------------
openai_llm = OpenAI(
    model_name="text-davinci-003", temperature=0.1, max_tokens=1000 # type: ignore
)  # max token length is 4097


#%%
# Create a PromptTemplate and LLMChain
# --------------------------------------------------------------
template_translation = """User: translate in english the text included between < and > in English

<{text}>

Assistant: I present you the translation in english, an nothing else, of the provided text
"""

#%%
# Run the LLMChain with Falcon7b: translate the text
# --------------------------------------------------------------
prompt = PromptTemplate(template=template_translation, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

question = f"""
{text_input}
"""
response = llm_chain.run(question)
translated_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(translated_text)

#%%
# Run the LLMChain with Falcon7b: understand the text
# --------------------------------------------------------------
template_extraction="""User: I need you to extract the soft skills from the job described in the text included between < and >

<{text}>

Assistant: The Soft skills are
"""

prompt = PromptTemplate(template=template_extraction, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

question = f"""
{translated_text}
"""
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)

#%%
# Run the LLMChain with OpenAI
# --------------------------------------------------------------
#%%
prompt = PromptTemplate(template=template_extraction, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=openai_llm)

question = f"""
{translated_text}
"""
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)



















################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################

# --------------------------------------------------------------
# Split text
# -------------------------------------------------------------
#%%

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
docs = text_splitter.split_text(TEXT)

# --------------------------------------------------------------
# Summarization with LangChain
# --------------------------------------------------------------
#%%
# Add map_prompt and combine_prompt to the chain for custom summarization
chain = load_summarize_chain(falcon_llm, chain_type="map_reduce", verbose=True)
print(chain.llm_chain.prompt.template)
print(chain.combine_document_chain.llm_chain.prompt.template)

# --------------------------------------------------------------
# Test the Falcon model with text summarization
# --------------------------------------------------------------
#%%
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)


# --------------------------------------------------------------
# Load an OpenAI model for comparison
# --------------------------------------------------------------
#%%
openai_llm = OpenAI(
    model_name="text-davinci-003", temperature=0.1, max_tokens=500
)  # max token length is 4097
chain = load_summarize_chain(openai_llm, chain_type="map_reduce", verbose=True)
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width=100, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)
