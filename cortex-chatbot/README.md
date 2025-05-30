# Example FOMC Chatbot using Cortex Search

This example demonstrates how to build a chatbot that uses the Cortex Search service to answer questions about the Federal Open Market Committee (FOMC) meeting minutes.  It is based on the snowflake example found [here](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/tutorials/cortex-search-tutorial-3-chat-advanced#introduction).

This folder containes the code for the chatbot in custom-copilot/ and it has some of the snowflake scripts used for setting up resources in snowflake.

## Setup

This project assumes that you have already setup your cortex search.  You should copy the .env.example file to .env and fill in the values.
```
cp .env.example .env
```
Then you can create your environment and install the dependencies.  I recommend using conda because of the ML dependencies.  And snowflake requires python < 3.12, so here we can use 3.11.  This repo provides both an environment.yml file and a requirements.txt file.  You can use either to create your environment.

```
conda env create -f environment.yml
conda activate cortex-env
```

Once the dependencies are installed, you can run the app.
```
uvicorn custom-copilot.main:app --port 7777 --reload 
```

This will start the FastAPI server on port 7777.  You can then add the custom agent to the OpenBB Copilot interface in the top left.  
