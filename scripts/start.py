import openai

# from parse_python_file import parse_python_file, unparse
# # # # # # # # # # # # # # # # # # # # # #
import ast

# Define a custom visitor class to traverse the AST and collect functions
class FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        self.functions.append(
            {
                "name": node.name,
                "full_text": ast.unparse(node),
            }
        )
        self.generic_visit(node)


# Parse the Python file and collect function names and full function text
def parse_python_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        source_code = file.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Error parsing file: {e}")
        return []

    visitor = FunctionVisitor()
    visitor.visit(tree)

    return visitor.functions


# # # # # # # # # # # # # # # # # # # # # #

# Load environment variables from .env file
# Retrieve the OpenAI API key from the environment variables
openai.api_key = "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"

# Define the function to request the summary
def generate_summary(file_path):

    print(file_path)
    functions = parse_python_file(file_path)
    for function in functions:
        print(
            "\n\n******************************\n",
            function["name"],
            "\n******************************\n",
        )
        print(function["full_text"])

        text = (
            "provide a review in markdown with 3 sections: Readability, Maintainability, and Performance. Write code samples of how it can be improved with code blocks that start with ```python.\n\n```\n"
            + function["full_text"]
            + "\n```\n"
        )

        try:
            # Adjust the parameters according to your requirements
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=text,
                max_tokens=1100,  # Adjust the summary length as needed
                temperature=0.3,  # Adjust the creativity level
                n=1,  # Generate a single summary
                stop=None,  # You can add a custom stop condition if desired
            )

            summary = response.choices[0].text.strip()
            # print("******************************\n\n", summary)

            # Save the summary as a Markdown file
            output_file = f"output/{function['name']}.md"
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(summary)

            # Print a success message
            print(f"Summary saved to {output_file} successfully.")

        except openai.error.AuthenticationError as e:
            print("OpenAI unknown authentication error")
            print(e.json_body)
            print(e.headers)


# Provide the path to the text file you want to summarize
file_path = "client.py"

# Call the function to generate the summary
generate_summary(file_path)
