import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage
from genai import Client, Credentials
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

with open("prompt.txt") as f:
    prompt_body = f.read()

# OpenAI Models
if os.environ.get('OPENAI_API_KEY') is not None:
  provider = "openai"
  models = ["gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4-1106-preview"
            ]

  prompt_template = """
  You are an AI Assistant trained on migrating enterprise JavaEE code to Quarkus.
  {prompt_body}
  """

  prompt = PromptTemplate.from_template(prompt_template)

  for model in models:
    llm = ChatOpenAI(temperature=0.1, model_name=model, streaming=True)
    chain = LLMChain(llm=llm, prompt=prompt)
    print("Starting model " + model)
    result = chain.invoke({"prompt_body": prompt_body})
    print("Finished model " + model)
  
    with open("output/" + provider + "_" + model + ".txt", "w") as file:
      file.write(result["text"])

# IBM Granite Models
if os.environ.get('GENAI_KEY') is not None:
  provider = "ibm"
  models = ["ibm/granite-13b-instruct-v1",
            "ibm/granite-13b-instruct-v2",
            "ibm/granite-20b-5lang-instruct-rc",
            "ibm/granite-20b-code-instruct-v1",
            "ibm/granite-20b-code-instruct-v1-gptq"
            ]

  prompt_template = """
  You are an AI Assistant trained on migrating enterprise JavaEE code to Quarkus.
  {prompt_body}
  """

  prompt = PromptTemplate.from_template(prompt_template)

  for model in models:
    llm = LangChainChatInterface(
      client=Client(credentials=Credentials.from_env()),
      model_id=model,
      parameters=TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=4096,
        min_new_tokens=10,
        temperature=0.1,
        top_k=50,
        top_p=1,
        return_options=TextGenerationReturnOptions(input_text=False, input_tokens=True),
      ),
      moderations=ModerationParameters(
        # Threshold is set to very low level to flag everything (testing purposes)
        # or set to True to enable HAP with default settings
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
      ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    print("Starting model " + model)
    result = chain.invoke({"prompt_body": prompt_body})
    print("Finished model " + model)

    model = model.replace("/", "_")
    with open("output/" + provider + "_" + model + ".txt", "w") as file:
      file.write(result["text"])

# IBM LLama Models
if os.environ.get('GENAI_KEY') is not None:
  provider = "ibm"
  models = ["ibm-mistralai/mixtral-8x7b-instruct-v01-q",
            "codellama/codellama-34b-instruct",
            "codellama/codellama-70b-instruct",
            "mistralai/mistral-7b-instruct-v0-2",
            "thebloke/mixtral-8x7b-v0-1-gptq"
            ]

  prompt_template = """
  <s>[INST] <<SYS>>
  You are an AI Assistant trained on migrating enterprise JavaEE code to Quarkus.
  <</SYS>>
  {prompt_body}
  [/INST]
  """
  
  prompt = PromptTemplate.from_template(prompt_template)

  for model in models:
    llm = LangChainChatInterface(
      client=Client(credentials=Credentials.from_env()),
      model_id=model,
      parameters=TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=4096,
        min_new_tokens=10,
        temperature=0.1,
        top_k=50,
        top_p=1,
        return_options=TextGenerationReturnOptions(input_text=False, input_tokens=True),
      ),
      moderations=ModerationParameters(
        # Threshold is set to very low level to flag everything (testing purposes)
        # or set to True to enable HAP with default settings
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
      ),
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    print("Starting model " + model)
    result = chain.invoke({"prompt_body": prompt_body})
    print("Finished model " + model)

    model = model.replace("/", "_")
    with open("output/" + provider + "_" + model + ".txt", "w") as file:
      file.write(result["text"])
