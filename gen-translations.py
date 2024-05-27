import outlines
from outlines import models, generate
from os import environ as env

gpt_4_0613 = models.azure_openai(
    model_name="gpt-4-0613",
    api_key=env["AZURE_API_KEY"],
    api_version=env["AZURE_API_VERSION"],
    azure_endpoint=env["AZURE_API_BASE"],
    deployment_name="gpt-4-vision-japaneasy",
)

gpt_35_turbo = models.azure_openai(
    model_name="gpt-3.5-turbo",
    api_key=env["AZURE_API_KEY"],
    api_version=env["AZURE_API_VERSION"],
    azure_endpoint=env["AZURE_API_BASE"],
    deployment_name="gpt-35-turbo-deployment",
)


@outlines.prompt
def greetings(name, question):
    """Hello, {{ name }}!
    {{ question }}
    """


prompt = greetings("user", "How are you?")

generator = generate.text(gpt_35_turbo)

answer = generator(prompt)
print(answer)
