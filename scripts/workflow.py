import openai
import logging
import sys
import requests

# from parse_python_file import parse_python_file, unparse
# # # # # # # # # # # # # # # # # # # # # #
import ast
# Provide the path to the text file you want to summarize
#file_path = "./client.py"

# Define a custom visitor class to traverse the AST and collect functions
class FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        self.functions.append({
            'name': node.name,
            'full_text': ast.unparse(node),
        })
        self.generic_visit(node)

# Parse the Python file and collect function names and full function text
def parse_python_file(file_path):
    print(file_path)
    with open(file_path, 'r') as file:
        source_code = file.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logging.info(f"Error parsing file: {e}")
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
    print("SUM")
    with open(file_path, "r", encoding="utf-8") as file:
        text = "provide a review in markdown with 3 sections: Readability, Maintainability, and Performance. Include code samples of how it can be improved.\n\n```" + file.read() + "```"

        functions = parse_python_file(file_path)
        for function in functions:
            print(f"******************************\n\n{function['full_text']}")

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
                print(f"******************************\n\n{summary}")

                # # Save the summary as a Markdown file
                # output_file = f"output/{function['name']}.md"
                # with open(output_file, "w", encoding="utf-8") as file:
                #     file.write(summary)

                #     # Print a success message
                #print(f"Summary saved to {output_file} successfully.")

            except openai.error.AuthenticationError as e:
                print("OpenAI unknown authentication error")
                print(e.json_body)
                print(e.headers)


def create_github_issue(user, head_branch, title, body,link):
    
    repo_owner="patprzybillacognizant" 
    repo_name="gptpoc"
    base_branch="main"
    
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    print(url)
    headers = {
        "Authorization": "Bearer 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",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "title": title + link,
        "body": body,
        "label": "AI Suggestions"
    }
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 201:
        pr_data = response.json()
        pr_number = pr_data.get("number")
        pr_url = pr_data.get("html_url")
        print(f"PR created successfully: {pr_url}")
        return pr_number
    else:
        print(response.status_code)
        print(f"Failed to create PR: {response.text}")
        return None
def render_github_markdown(body):
    url = "https://api.github.com/markdown"
    print(url)
    headers = {
        "Authorization": "Bearer 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",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "text": body,
    }
    response = requests.post(url, headers=headers, json=payload)
    
def main():
    print('cmd entry:', sys.argv)
    logging.info("main()")
    print("main()")
    body = generate_summary(sys.argv[1])
    print(body)
    html = render_github_markdown(body)
    pr_number = create_github_issue(sys.argv[2], sys.argv[3], sys.argv[4], html, "asdasd" )
    print(pr_number)
    

if __name__ == "__main__":
    main()
