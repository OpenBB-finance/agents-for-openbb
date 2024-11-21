# Bring your own Copilot to the OpenBB app

Welcome to the example repository for integrating custom copilots into the OpenBB app.

This repository provides everything you need to build and add your own custom
copilots that are compatible with the OpenBB app.

Here are a few common reasons why you might want to build your own copilot:
- You have a unique data source that you don't want to add as a custom integration to OpenBB.
- You want to use a specific LLM.
- You want to use a local LLM.
- You want a Copilot that is self-hosted on your infrastructure.
- You are running on-premise in a locked-down environment that doesn't allow data to leave your VPC.


## Overview

To integrate a custom Copilot that you can interact with from the OpenBB app,
you'll need to create a backend API that the OpenBB app can make requests to.  


Your custom copilot API will respond with Server-Sent Events
([SSEs](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)).

**Note: If you're looking to get started
quickly, we suggest running one of the example copilots included as part of
this repository, and adding it as a custom copilot to the OpenBB app (each example copilot includes instructions on how to run them). Cloning and modifying an example copilot is a great way to build a custom copilot.**

## The Copilot protocol is stateless

The most important concept to understand is that the copilot protocol is
_stateless_.  This means that every request from the OpenBB app to your copilot
will include all previous messages (such as AI completions, human messages,
function calls, and function call results) in the request payload.

This means it is not necessary for your custom copilot to maintain any state
between requests. It should simply use the request payload to generate a response.

The OpenBB app is solely responsible for maintaining the conversation state, and will
append the responses to the `messages` array in the request payload. 

## Handling requests from the OpenBB app

The OpenBB app will make POST requests to the `query` endpoint defined in your
`copilots.json` file (more on this later). The payload of this request will
contain data such as the current conversation's messages, any explicitly-added
context, information about widgets on the currently-active dashboard, URLs to
retrieve, and so on.

### API Request Schema

The core of the query request schema you must implement is as follows:

```python
{
  "messages": [  # <-- the chat messages between the user and the copilot (including function calls and results)
    {
      "role": "human",  # <-- each message has a role: "human", "ai", or "tool"
      "content": "Hi there."  # <-- the content of the message
    }
  ],
  "context": [  # <-- explicitly added context by the user (optional)
    {
      "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",  # <-- the UUID of the widget
      "name": "<widget name>",  # <-- the name of the widget
      "description": "<widget description>",  # <-- the description of the widget
      "data": {
        "content": "<data>"  # <-- the data of the widget
      },
      "metadata": {
        "<metadata key>": "<metadata value>",  # <-- the metadata of the widget
        ...
      }
    },
    ...
  ],
  "widgets": [  # <-- the widgets currently visible on the active dashboard on the OpenBB app (optional)
    {
      "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",  # <-- the UUID of the widget
      "name": "<widget name>",  # <-- the name of the widget
      "description": "<widget description>",  # <-- the description of the widget
      "metadata": {
        "<metadata key>": "<metadata value>",  # <-- the metadata of the widget
        ...
      }
    },
    ...
  ],
}
```

We'll go over each of these fields in more detail below.

#### `messages`
This is the list of messages between the user and the copilot. This includes the
user's messages, the copilot's messages, function calls, and function call
results. Each message has a `role` and `content`.

The simplest example is when no function calling is involved, which simply
consists of an array of `human` and `ai` messages.

The OpenBB app automatically appends all return `ai` messages (from your Copilot)
to the `messages` array of any follow-up request.

```python
# Only one human message
{
  "messages": [
    {
      "role": "human", 
      "content": "Hi there."
    }
  ],
  ...
}
```

```python
# Multiple messages
{
  "messages": [
    {
      "role": "human", 
      "content": "Hi there."
    },
    {
      "role": "ai", 
      "content": "Hi there, I'm a copilot. How are you?"
    },
    {
      "role": "human", 
      "content": "I'm fine, thank you. What is the weather in Tokyo?"
    }
  ],
  ...
}
```

Function calls to the OpenBB app (such as when retrieving widget data), as well as the results of those function calls (containing the widget data), are also included in the `messages` array. For information on function calling, see the "Function Calling" section below.

#### `context`

This is an optional array of widget data that will be sent by the OpenBB app
when widgets have been added as context explicitly by the user. This happens
when the user clicks on the "Add as Context" button on a widget in the OpenBB app.

<img src="https://github.com/user-attachments/assets/6c84dc5b-7615-4483-a956-67ac526d5800" alt="explicit_context" width="70%"/>

The `context` field works as follows:

```python
{
  ...
  "context": [
    {
      "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",  # <-- each widget has a UUID
      "name": "Analyst Estimates",
      "description": "Contains analyst estimates for a ticker",
      "data": {
        "content": "<data>"  # <-- the data of the widget could either be a JSON string or plaintext (you must choose how to handle this in your copilot)
      },
      "metadata": {  # <-- additional metadata about the widget
        "symbol": "AAPL",
        "period": "quarter",
        "source": "Financial Modelling Prep",
        "lastUpdated": 1728998071322
      }
    },
    {
      "uuid": "8b2e5f79-3a1d-4c9e-b6f8-1e7d2a9c0b3d",  # <-- the context can have multiple widgets
      "name": "Earnings Transcripts",  
      "description": "Contains earnings transcripts for a ticker",
      "data": {
        "content": "<data>"  # <-- the data of the widget
      },
      "metadata": {
        "symbol": "AAPL",
        "period": "quarter",
        "source": "Intrinio",
        "lastUpdated": 1728998071322
      }
    },
    ...
  ],
  ...
}
```

#### `widgets`

This is an array of widgets that are currently visible on the active dashboard
on the OpenBB app.
**This is only useful if you're planning on implementing function calling in
*your custom copilot** (which is recommended, but not required), which
allows it to request widget data from the user's currently-active dashboard on
the OpenBB app.

```python
{
  ...
  "widgets": [
    {
      "uuid": "c276369e-e469-4689-b5fe-3f8c76f7c45a",
      "name": "Stock Price Quote Widget",
      "description": "Contains the current stock price of a ticker",
      "metadata": {
        "ticker": "AAPL"
      }
    },
    {
      "uuid": "9f8e7d6c-5b4a-3c2e-1d0f-9e8d7c6b5a4b",
      "name": "Financial Ratios Widget",
      "description": "Displays key financial ratios for a company",
      "metadata": {
        "ticker": "AAPL",
        "period": "TTM"
      }
    },
    ...
  ],
  ...
}
```

## Responding to the OpenBB app

Your custom copilot must respond to the OpenBB app's request using a variety of Server-Sent Events (SSEs).

The OpenBB app can process the following SSEs:

- `copilotMessageChunk`: Used to return streamed copilot tokens (partial
responses) back to the OpenBB app. These responses can be streamed as they are
generated.
- `copilotMessageArtifact`: Used to return an artifact to the OpenBB app as part
of the Copilot's response.  This allows your copilot to return tables, charts,
and long-form text excerpts that will be rendered by the OpenBB app. Often
interleaved the `copilotMessageChunk` SSEs.
- `copilotCitationCollection`: Used to return a collection of citations back to the OpenBB app. This is useful for returning structured data, such as a list of news articles, research reports, or other sources that were used to generate the Copilot's response. This should be returned after the `copilotMessageChunk` SSEs have finished streaming.
- `copilotFunctionCall`: Used to request data (e.g., widget data) or perform a
specific function. This instructs the OpenBB app to take further action on the
client's side. This is only necessary if you're planning on implementing
function calling in your custom copilot.
- `copilotStatusUpdate`: Used to send status updates or "reasoning steps" back to
the OpenBB app. These are user-friendly "updates" that are displayed in the
copilot window of the OpenBB app, and are useful for informing the user about what your custom copilot is doing under-the-hood.

#### `copilotMessageChunk`
The message chunk SSE has the following format:

```
event: copilotMessageChunk
data: {"delta":"H"}  # <-- the `data` field must be a JSON object.
```
The `delta` must be a string, but can be of any length. We suggest streaming
back each chunk you receive from your LLM as soon as it's generated as a `delta`.

For example, if you wanted to stream back the message "Hi!", you would send the
following SSEs:

```
event: copilotMessageChunk
data: {"delta":"H"}

event: copilotMessageChunk
data: {"delta":"i"}

event: copilotMessageChunk
data: {"delta":"!"}
```

#### `copilotMessageArtifact`
The message artifact SSE has the following format:

```
event: copilotMessageArtifact
data: {"type": "<type>", "uuid": "<uuid>", "content": "<content>", "chart_params": <chart_params>}
```

An artifact can be a table, chart, or long-form text excerpt.
Let's go over each of these with examples:

A text artifact:

```
{
    "type": "text",
    "uuid": "123e4567-e89b-12d3-a456-426614174000",
    "content": "This is a sample text content",
}
```



#### `copilotFunctionCall` (only required for function calling)
The function call SSE has the following format:

```
event: copilotFunctionCall
data: {"function":"get_widget_data","input_arguments":{"widget_uuid":"c276369e-e469-4689-b5fe-3f8c76f7c45a"}}
```

Again, the `data` field must be a JSON object. The `function` field is the name
of the function to be called (currently only `get_widget_data` is supported),
and the `input_arguments` field is a dictionary of arguments to be passed to the
function. For the `get_widget_data` function, the only required argument is
`widget_uuid`, which is the UUID of the widget to retrieve data for (from one of
the UUIDs in the `widgets` array of the request).



## Function Calling

By adding function calling to your copilot, it will be able to request data that
is visible on a user's currently-active dashboard in the OpenBB app.

A list of all widgets currently visible on a user's dashboard is sent to your
copilot in the `widgets` array of the request payload.

To retrieve the data from a widget, your copilot should respond with a
`copilotFunctionCall` event, specifying the widget UUID:

```
event: copilotFunctionCall
data: {"function":"get_widget_data","input_arguments":{"widget_uuid":"c276369e-e469-4689-b5fe-3f8c76f7c45a"}}
```

After emitting a `copilotFunctionCall` event, you must close the connection and wait for a new query request from the OpenBB app.

When a `copilotFunctionCall` event is received, the OpenBB app will retrieve
the data, and initiate a **new** query request. This new query request will
include the original function call, as well as the function call result in the
`messages` array.

```python
{
  ...
  "messages": [
    ...
    {
      "role": "ai",
      "content": "{\"function\":\"get_widget_data\",\"input_arguments\":{\"widget_uuid\":\"c276369e-e469-4689-b5fe-3f8c76f7c45a\"}}"
    },
    {
      "role": "tool",
      "function": "get_widget_data",
      "data": {
        "content": "<data>"
      } 
    }
  ]
}
```

Notice that:
- Both the function call and the function call result are included in the `messages` array. 
- The `content` field of the function call `ai` message is a verbatim string-encoded JSON object of the `data` field of the `copilotFunctionCall` event (this is a very useful mechanism for smuggling additional metadata related to the function call, if your copilot needs it).

Currently, the only function call supported by the OpenBB app is `get_widget_data`, which retrieves data from a specific widget.

### Function call example

Your custom copilot receives the following request from the OpenBB app:

```json
{
  "messages": [
    {
      "role": "human",
      "content": "What is the current stock price of AAPL?"
    }
  ],
  "widgets": [
    {
      "uuid": "38181a68-9650-4940-84fb-a3f29c8869f3",
      "name": "Historical Stock Price",
      "description": "Historical Stock Price",
      "metadata": {
        "symbol": "AAPL",
        "source": "Financial Modelling Prep",
        "lastUpdated": 1728994470324
      }
    }
  ]
}
```

You then parse the response, format the messages to your LLM (including information on which widgets are available).  Let's assume
that your copilot determines that the user's query can be answered using the widget available, and generates a function call to retrieve the data.

Your copilot then responds with the following SSE and close the connection:

```
event: copilotFunctionCall
data: {"function":"get_widget_data","input_arguments":{"widget_uuid":"38181a68-9650-4940-84fb-a3f29c8869f3"}}
```

The OpenBB app will then execute the specified function, and make a new query request to your custom copilot:

```python
{
  "messages": [
    {
      "role": "human",
      "content": "What is the current stock price of AAPL?"
    },
    {
      "role": "ai",
      "content": "{\"function\":\"get_widget_data\",\"input_arguments\":{\"widget_uuid\":\"38181a68-9650-4940-84fb-a3f29c8869f3\"}}"  
    },
    {
      "role": "tool",
      "function": "get_widget_data",
      "data": {
        "content": "[{\"date\":\"2024-10-15T00:00:00-04:00\",\"open\":233.61,\"high\":237.49,\"low\":232.37,\"close\":233.85,\"volume\":61901688,\"vwap\":234.33,\"adj_close\":233.85,\"change\":0.24,\"change_percent\":0.0010274},{\"date\":\"2024-10-14T00:00:00-04:00\",\"open\":228.7,\"high\":231.73,\"low\":228.6,\"close\":231.3,\"volume\":39882100,\"vwap\":230.0825,\"adj_close\":231.3,\"change\":2.6,\"change_percent\":0.0114},{\"date\":\"2024-10-11T00:00:00-04:00\",\"open\":229.3,\"high\":233.2,\"low\":228.9,\"close\":231.0,\"volume\":32581944,\"vwap\":231.0333,\"adj_close\":231.0,\"change\":1.7,\"change_percent\":0.0074}, ... ]"
      }
    } 
  ],
  "widgets": [
    {
      "uuid": "38181a68-9650-4940-84fb-a3f29c8869f3",
      "name": "Historical Stock Price",
      "description": "Historical Stock Price",
      "metadata": {
        "symbol": "AAPL",
        "source": "Financial Modelling Prep",
        "lastUpdated": 1728994470324
      }
    }
}
```


You then parse the response, process the data, and format the messages to your LLM. Let's assume that the LLM then generates a string of tokens to answer the user's query. These are then streamed back to the user using the `copilotMessageChunk` SSE:

```
event: copilotMessageChunk
data: {"delta":"The"}

event: copilotMessageChunk
data: {"delta":" current"}

event: copilotMessageChunk
data: {"delta":" stock"}

event: copilotMessageChunk
data: {"delta":" price"}

event: copilotMessageChunk
data: {"delta":" of"}

event: copilotMessageChunk
data: {"delta":" Apple"}

event: copilotMessageChunk
data: {"delta":" Inc."}

event: copilotMessageChunk
data: {"delta":" (AAPL)"}

event: copilotMessageChunk
data: {"delta":" is"}

event: copilotMessageChunk
data: {"delta":" $150.75."}
```


## Configuring your custom copilot for the OpenBB app (`copilots.json`)

To integrate your custom copilot with the OpenBB app, you need to configure and
serve a `copilots.json` file. This file defines how your custom copilot connects
with the frontend, including which features are supported by your custom
copilot, and where requests  should be sent.  

Here is an example copilots.json configuration:

```python
{
  "example_copilot": { # <-- the ID of your copilot
    "name": "Mistral Example Co. Copilot", # <-- the display name of your copilot
    "description": "AI-powered financial copilot that uses Mistral Large as its LLM.", # <-- a short description of your copilot
    "image": "<url>", # <-- a URL to an image icon for your copilot
    "hasStreaming": true, # <-- whether your copilot supports streaming responses via SSEs. This must always be true.
    "hasFunctionCalling": true, # <-- whether your copilot supports function calling
    "endpoints": {
      "query": "<url>" # <-- the URL that the OpenBB app will send requests to. For example, "http://localhost:7777/v1/query"
    }
  }
}
```

Your `copilots.json` file must be served at `<your-host>/copilots.json`, for example, `http://localhost:7777/copilots.json`.