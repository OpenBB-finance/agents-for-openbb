import os

import dotenv
from snowflake.core import Root
from snowflake.cortex import complete
from snowflake.snowpark import Session

dotenv.load_dotenv()

CONNECTION_PARAMETERS = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.environ["SNOWFLAKE_ROLE"],
}

session = Session.builder.configs(CONNECTION_PARAMETERS).create()

root = Root(session)
# Assuming we only have one service
# IF YOU HAVE MULTIPLE SERVICES, YOU WILL NEED TO CHANGE THIS CODE TO GET THE CORRECT SERVICE
services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
database, warehouse, schema, service_name = (
    services[0].database_name,
    services[0].warehouse,
    services[0].schema_name,
    services[0].name,
)


def query_cortex_search_service(
    query,
    columns=[],
    filter={},
    limit=5,
    database=database,
    schema=schema,
    service_name=service_name,
):
    cortex_search_service = (
        root.databases[database].schemas[schema].cortex_search_services[service_name]
    )
    context_documents = cortex_search_service.search(
        query, columns=columns, filter=filter, limit=limit
    )
    return context_documents.results


def create_fomc_prompt(
    query, columns=["chunk", "file_url", "relative_path"], filter={}, limit=5
):
    """
    Create a prompt for the language model by combining the query with context retrieved
    from the cortex search service.

    Args:
        query (str): The user's question to generate a prompt for.
        columns (list): Columns to retrieve from the search service
        filter (dict): Filter criteria for the search
        limit (int): Maximum number of results to return

    Returns:
        tuple: (str, list) The generated prompt and search results
    """
    # Get context from Cortex search service
    if not filter:
        filter = {"@and": [{"@eq": {"language": "English"}}]}

    results = query_cortex_search_service(
        query, columns=columns, filter=filter, limit=limit
    )

    # Create the prompt template
    prompt = f"""
    [INST]
    You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question,
    you will also be given context provided between <context> and </context> tags. Use that context
    to provide a summary that addresses the user's question. Ensure the answer is coherent, concise,
    and directly relevant to the user's question.

    If the user asks a generic question which cannot be answered with the given context, or if the user asks a question that is not related to the context,
    just say "I don't know the answer to that question."

    Don't say things like "according to the provided context".  But do cite any specific context so that the user can find more information.  No need to respond with new lines or characters that change formatting.  Please cite your source inline using a format like (Citation : [])

    <context>
    {results}
    </context>
    <question>
    {query}
    </question>
    [/INST]
    Answer:
    """
    return prompt, results


def llm_complete(model, prompt):
    """
    Generate a completion for the given prompt using the specified model.

    Args:
        model (str): The name of the model to use for completion.
        prompt (str): The prompt to generate a completion for.

    Returns:
        str: The generated completion.
    """
    return complete(model, prompt, stream=True).replace("$", "\$")


def _llm(query):
    prompt, results = create_fomc_prompt(query)
    response = llm_complete("mistral-large", prompt)
    for chunk in response:
        yield chunk.replace("$", "\$").replace("   ", " ")
    # After response is complete, yield the references table
    markdown_table = "\n\n\n###### Search Results \n\n| PDF Title |\n|-------|-----|\n"
    for ref in results:
        markdown_table += f"| {ref['relative_path']}| \n"

    # Yield the table as a final chunk
    yield markdown_table
