from langchain_core.prompts import PromptTemplate
from langchain_core.load import dumps

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}"
with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
   - Include relevant mathematical equations if present.
   - Explain mathematical concepts with intuitive examples
     and code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If information is unavailable, state:
"Insufficient information available."

Ensure the summary follows the requested style and length.
""",
    input_variables=[
        "paper_input",
        "style_input",
        "length_input",
    ],
    validate_template=True,
)

with open("template.json", "w", encoding="utf-8") as file:
    file.write(dumps(template, pretty=True))

print("template.json generated successfully!")