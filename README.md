# The Complete LangChain Guidebook
## From Zero to Advanced: Your Definitive Resource

---

# Table of Contents

1. [Foundation: Understanding LangChain](#part-1-foundation)
2. [Environment Setup](#part-2-setup)
3. [Beginner Level: First Steps](#part-3-beginner)
4. [Intermediate Level: Building Real Applications](#part-4-intermediate)
5. [Advanced Level: Production-Ready Systems](#part-5-advanced)
6. [Complete Projects](#part-6-projects)
7. [Best Practices & Optimization](#part-7-best-practices)

---

# Part 1: Foundation - Understanding LangChain {#part-1-foundation}

## What is LangChain and Why Does It Exist?

Before we write a single line of code, let's understand the fundamental problem that LangChain solves. Imagine you're building an intelligent assistant that helps students study. Your assistant needs to:

- Remember what the student asked five questions ago
- Search through the student's textbooks to find relevant information
- Break down complex topics into simpler explanations
- Decide whether to search the web, check the textbook, or use its own knowledge

If you try to build this using just the raw OpenAI or Anthropic API, you'll quickly find yourself writing hundreds of lines of plumbing code: managing conversation history, formatting prompts, handling errors, orchestrating multiple API calls, and more. You'll spend more time on infrastructure than on your actual application logic.

**This is exactly the problem LangChain solves.** It provides pre-built, battle-tested components for all the common patterns you need when building LLM applications, so you can focus on what makes your application unique instead of reinventing the wheel.

## The Mental Model: Components as LEGO Blocks

Think of LangChain as a box of specialized LEGO blocks. Each block does one thing well, and you can snap them together to build complex structures. Here's the key insight: **a sophisticated LLM application is just simple components connected in the right way**.

Let me show you the flow visually:

```
Simple Application (Single LLM Call):
User Question → LLM → Answer

With Memory (Chatbot):
User Question → [Memory Loads History] → LLM → [Memory Saves Exchange] → Answer

With Retrieval (Document Q&A):
User Question → [Search Documents] → [Inject Relevant Chunks] → LLM → Answer

With Agents (Smart Assistant):
User Question → Agent → [Decides: Search Web or Calculate or Use Tool] → LLM → Answer
```

Notice how we're building complexity by adding layers, not by making everything more complicated. This composability is LangChain's superpower.

## The Six Core Components Explained

Let me walk you through each core component with real-world analogies and explain exactly when and why you'd use each one.

### 1. Models (LLMs and Chat Models)

**What they are**: Models are your interface to the actual language models like GPT-4, Claude, Llama, or Gemini. LangChain wraps these APIs in a consistent interface so you can switch between providers without rewriting your code.

**The analogy**: Think of models like different car engines. A Toyota engine and a BMW engine do the same fundamental thing (make the car move), but they have different quirks, different fuel requirements, and different interfaces. LangChain is like a universal transmission that works with any engine, so you can swap engines without redesigning your entire car.

**When to use**: Always. Every LangChain application needs a model. It's the brain of your system.

**Two types you need to know**:

**LLMs** are for single-turn text completion. You give them a prompt, they complete it. Think of old-school GPT-3 where you'd say "The capital of France is" and it continues with "Paris". These are becoming less common as chat models improve.

**Chat Models** are designed for conversations. They understand messages with roles like user, assistant, and system. This is what you use for almost everything today (GPT-4, Claude, etc.).

### 2. Prompts and Prompt Templates

**What they are**: Prompts are the instructions you give to the LLM. Prompt Templates are prompts with variables that you can fill in dynamically. Instead of writing a new prompt every time, you create a template once and reuse it with different inputs.

**The analogy**: Think of a prompt template like a form letter. You have a template that says "Dear {name}, Thank you for your purchase of {product}. Your order will arrive on {date}." Instead of writing a new letter each time, you just fill in the blanks.

**When to use**: Whenever you find yourself writing similar prompts over and over, or when your prompts need to include dynamic user data.

**Why this matters**: Good prompting is an art. Once you craft a good prompt, you want to reuse it consistently. Templates let you maintain quality while scaling.

### 3. Chains

**What they are**: Chains connect multiple components in sequence. The output of one step becomes the input to the next. You can chain together prompts, models, data retrieval, and transformations.

**The analogy**: Think of chains like a factory assembly line. Raw materials (user input) enter at one end, pass through various stations (prompt formatting, LLM call, output parsing), and finished products (final answer) come out the other end.

**When to use**: When your application needs multiple steps. For example: summarize a document, then translate the summary, then extract key points from the translation. That's a three-step chain.

**Common patterns**:
- Sequential chains where each step feeds the next
- Parallel chains where multiple operations happen simultaneously
- Conditional chains where the flow depends on intermediate results

### 4. Memory

**What it is**: Memory stores conversation history and context so the LLM can reference earlier parts of the conversation. Without memory, the LLM has amnesia and treats every question as if it's the first one.

**The analogy**: Imagine calling customer support and every time you explain your problem, the agent says "I have no record of previous calls." Frustrating, right? Memory is what lets the agent say "I see you called about this yesterday, let me continue where we left off."

**When to use**: Any time you're building a chatbot or conversational interface where context matters. If the user says "What's the weather?" and then asks "How about tomorrow?", you need memory to know they're still talking about weather.

**Types of memory**:
- **Buffer Memory**: Stores the last N messages (simple but can get long)
- **Summary Memory**: Summarizes old conversations to save space
- **Entity Memory**: Tracks specific entities mentioned (people, places, things)
- **Knowledge Graph Memory**: Builds relationships between concepts

### 5. Retrievers and Vector Stores

**What they are**: When you have lots of documents (manuals, articles, books), you can't feed everything to the LLM at once. Retrievers search through your documents and find the most relevant pieces for the current query. Vector stores are specialized databases that enable semantic search (finding documents by meaning, not just keywords).

**The analogy**: Think of a library. You don't read every book when you have a question. You ask the librarian (retriever) and they find the relevant books. They don't just match keywords; they understand what you're really asking about. That's semantic search.

**When to use**: When you're building document Q&A systems, knowledge bases, or any application where the LLM needs to reference specific information that wasn't in its training data.

**How it works conceptually**:
1. You split documents into chunks
2. Each chunk gets converted to a vector (a list of numbers representing its meaning)
3. When a user asks a question, their question also becomes a vector
4. The system finds the chunks whose vectors are most similar (closest in meaning)
5. These relevant chunks get fed to the LLM along with the user's question

### 6. Agents and Tools

**What they are**: Agents are LLMs that can decide which tools to use to answer a question. Tools are functions the LLM can call (like a calculator, web search, database query, or API call). The LLM examines the question, decides what tools it needs, calls them, and synthesizes the results.

**The analogy**: Think of an agent like a personal assistant who knows when to look things up on Google, when to check your calendar, when to do a calculation, or when to just answer from their own knowledge. They're not just responding to commands; they're figuring out what needs to be done and doing it.

**When to use**: When your application needs to take actions or access information that the LLM doesn't know. For example, checking stock prices (API call), doing precise math (calculator), or finding current news (web search).

**The agent loop**:
1. User asks a question
2. Agent thinks about what it needs to do
3. Agent decides to use a tool and calls it
4. Tool returns a result
5. Agent incorporates the result and either answers or uses another tool
6. Repeat until the agent has enough information to answer

This is the most powerful pattern in LangChain because it gives the LLM agency to solve problems autonomously.

## Understanding the Flow: A Complete Example

Let me show you how all these pieces work together in a real application. Imagine you're building a customer support bot for an online store.

**The architecture**:
```
Customer Question: "What's the status of order #12345?"
         ↓
[Memory] - Loads conversation history
"Customer previously asked about shipping policies"
         ↓
[Agent Reasoning] - "I need to check order status"
         ↓
[Tool: Database Query] - Looks up order #12345
Result: "Order shipped on Oct 1, arriving Oct 5"
         ↓
[Memory] - Loads shipping policy from vector store
"Standard shipping takes 3-5 business days"
         ↓
[Prompt Template] - Formats everything nicely
"Given the order status and shipping policy, respond helpfully"
         ↓
[LLM] - Generates response
         ↓
[Memory] - Saves this exchange
         ↓
Response: "Your order shipped on October 1st and will arrive by October 5th. 
This is within our standard 3-5 business day shipping window. Is there 
anything else I can help you with?"
```

Notice how each component plays a specific role, and together they create an intelligent, context-aware system.

## Before vs After LangChain: A Detailed Comparison

Let me show you the difference with a concrete example. We'll build a simple chatbot that remembers conversation history.

**WITHOUT LangChain** (70+ lines of manual management):

```python
import openai
import json
from datetime import datetime

class ManualChatbot:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.conversation_history = []
        self.max_history = 10  # Limit to prevent token overflow
        
    def add_message(self, role, content):
        """Manually add messages to history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Manually trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def format_history_for_api(self):
        """Manually format history for OpenAI API"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]
    
    def chat(self, user_message):
        """Handle a chat message"""
        try:
            # Manually add user message
            self.add_message("user", user_message)
            
            # Manually prepare the API call
            messages = self.format_history_for_api()
            
            # Call the API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            # Manually extract response
            assistant_message = response.choices[0].message.content
            
            # Manually add assistant response
            self.add_message("assistant", assistant_message)
            
            return assistant_message
            
        except Exception as e:
            # Manually handle errors
            print(f"Error: {e}")
            return "Sorry, something went wrong."
    
    def clear_history(self):
        """Manually clear conversation history"""
        self.conversation_history = []
    
    def save_history(self, filename):
        """Manually save history to file"""
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_history(self, filename):
        """Manually load history from file"""
        try:
            with open(filename, 'r') as f:
                self.conversation_history = json.load(f)
        except FileNotFoundError:
            print("No history file found")

# Usage
bot = ManualChatbot(api_key="your-key")
print(bot.chat("Hi, my name is Alice"))
print(bot.chat("What's my name?"))
```

**WITH LangChain** (15 lines of clean code):

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # Shows you what's happening
)

# Usage - memory is automatic!
print(conversation.predict(input="Hi, my name is Alice"))
print(conversation.predict(input="What's my name?"))

# Save/load history is built in
conversation.memory.save_context({"input": "test"}, {"output": "test"})
```

See the difference? LangChain handles all the plumbing: history management, API formatting, error handling, token counting, and more. You just configure what you want and let it work.

## When Should You Use LangChain?

**Use LangChain when you need:**
- Multiple LLM calls in sequence (chains)
- Conversation memory/context
- Document retrieval and search
- Agents that use tools
- Switching between LLM providers
- Production features like logging, monitoring, streaming

**Don't use LangChain when:**
- You only need one simple LLM call (the provider's SDK is simpler)
- You need ultra-fine-grained control over every API parameter
- Your workflow is so unique that LangChain's abstractions don't fit
- You're learning LLMs for the first time (start with raw API, then graduate to LangChain)

**Rule of thumb**: If your application has more than 2-3 LLM calls or needs memory/tools/retrieval, LangChain will save you significant time.

---

# Part 2: Environment Setup {#part-2-setup}

## Installing LangChain

LangChain is modular, meaning you install only what you need. Let me walk you through the installation options.

**Basic installation** (just the core):
```bash
pip install langchain
```

**For OpenAI models**:
```bash
pip install langchain-openai
```

**For Anthropic (Claude) models**:
```bash
pip install langchain-anthropic
```

**For local/open-source models**:
```bash
pip install langchain-community
```

**For vector stores and embeddings**:
```bash
pip install langchain-chroma  # Chroma vector store
pip install langchain-pinecone  # Pinecone vector store
```

**All-in-one installation** (gets most common packages):
```bash
pip install langchain langchain-openai langchain-community langchain-chroma
```

**For this guide, I recommend**:
```bash
pip install langchain langchain-openai langchain-anthropic langchain-community chromadb
```

## Setting Up API Keys

You'll need API keys from LLM providers. Here's how to get and configure them.

**Getting API Keys:**

For OpenAI (GPT models):
1. Go to platform.openai.com
2. Sign up or log in
3. Go to API keys section
4. Create new key
5. Copy it immediately (you won't see it again)

For Anthropic (Claude models):
1. Go to console.anthropic.com
2. Sign up or log in
3. Go to API keys
4. Create key
5. Copy it

**Configuring Keys** (three methods):

**Method 1: Environment Variables** (recommended for security):
```bash
# On Mac/Linux, add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# On Windows, use Command Prompt
setx OPENAI_API_KEY "sk-..."
setx ANTHROPIC_API_KEY "sk-ant-..."
```

**Method 2: .env File** (good for local development):
```bash
# Create a .env file in your project directory
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Then in Python
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file
```

**Method 3: Direct in Code** (not recommended for production):
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

## Setting Up Your First Project

Create a project structure that will serve you well:

```
my_langchain_project/
├── .env                 # API keys (never commit this!)
├── .gitignore          # Ignore .env and other sensitive files
├── requirements.txt    # List of dependencies
├── main.py            # Your main code
├── chains/            # Custom chains
├── tools/             # Custom tools
└── data/              # Documents and data files
```

**requirements.txt example**:
```
langchain==0.1.0
langchain-openai==0.0.5
langchain-anthropic==0.1.1
langchain-community==0.0.20
chromadb==0.4.22
python-dotenv==1.0.0
```

**.gitignore example**:
```
.env
__pycache__/
*.pyc
.vscode/
.idea/
venv/
*.log
```

## Your First LangChain Script

Let's verify everything is working with a simple test script:

```python
# test_setup.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment")
    exit(1)

print("API key found!")

# Test LLM
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke("Say 'LangChain setup successful!' and nothing else")
    print(response.content)
    print("\n✅ Setup complete! You're ready to learn LangChain.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your API key and internet connection")
```

Run it:
```bash
python test_setup.py
```

If you see "LangChain setup successful!", you're ready to continue!

---

# Part 3: Beginner Level - First Steps {#part-3-beginner}

## Lesson 1: Your First LLM Call

Let's start with the absolute basics: calling an LLM through LangChain.

```python
from langchain_openai import ChatOpenAI

# Create an LLM instance
# temperature controls randomness (0 = deterministic, 1 = creative)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=100  # Limit response length
)

# Call the LLM
response = llm.invoke("What is LangChain in one sentence?")

# The response is a message object
print(response.content)  # The actual text
print(type(response))    # AIMessage object
```

**Understanding the code:**
- We create an LLM instance with configuration (model, temperature, etc.)
- We call `invoke()` with our question
- We get back an AIMessage object that contains the response

**Key parameters explained:**
- `temperature`: Controls randomness. 0 is deterministic (always same answer), 1 is creative (varied answers)
- `max_tokens`: Limits how long the response can be (saves money and ensures conciseness)
- `model`: Which specific model to use

## Lesson 2: Chat Models vs LLMs

There are two types of models in LangChain. Let me explain the difference.

**LLMs (Legacy)** - Text completion:
```python
from langchain.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
response = llm.invoke("The capital of France is")
print(response)  # Output: " Paris"
```

This is the old-school approach. You give it a prompt and it completes the text.

**Chat Models (Modern)** - Conversation-aware:
```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    SystemMessage(content="You are a helpful assistant that speaks like Shakespeare"),
    HumanMessage(content="Tell me about Python programming")
]

response = chat.invoke(messages)
print(response.content)
```

**Why Chat Models are better:**
- They understand conversational context
- You can set a system prompt (personality/instructions)
- They're designed for back-and-forth interaction
- Most modern LLMs are chat models

**Use chat models for everything going forward.** LLMs are mostly legacy.

## Lesson 3: Prompt Templates - Making Reusable Prompts

Hard-coding prompts is tedious. Prompt templates let you create reusable prompt formats with variables.

**Basic template:**
```python
from langchain.prompts import PromptTemplate

# Create a template with one variable
template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}"
)

# Use the template
prompt = template.format(topic="artificial intelligence")
print(prompt)
# Output: "Write a short poem about artificial intelligence"

# Now call the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
response = llm.invoke(prompt)
print(response.content)
```

**Multiple variables:**
```python
template = PromptTemplate(
    input_variables=["language", "concept"],
    template="Explain {concept} to a beginner programmer who only knows {language}"
)

prompt = template.format(language="Python", concept="recursion")
# Output: "Explain recursion to a beginner programmer who only knows Python"
```

**Why templates matter:**
- You can reuse good prompts
- You can test different variables easily
- You can maintain consistent formatting
- You separate data from instructions

## Lesson 4: Chat Prompt Templates - For Conversational Apps

Chat models work better with structured messages. LangChain has special templates for this.

```python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Create a chat template with system and user messages
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a {role} who explains things in a {style} way"
    ),
    HumanMessagePromptTemplate.from_template(
        "Explain {topic} to me"
    )
])

# Format the template
messages = chat_template.format_messages(
    role="teacher",
    style="simple and friendly",
    topic="quantum computing"
)

# Call the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
response = llm.invoke(messages)
print(response.content)
```

**Even simpler syntax:**
```python
from langchain.prompts import ChatPromptTemplate

# Use a shorthand
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in {domain}"),
    ("user", "{question}")
])

messages = template.format_messages(
    domain="history",
    question="Who was the first Roman emperor?"
)

response = llm.invoke(messages)
print(response.content)
```

## Lesson 5: Your First Chain - Connecting Components

Now we'll connect a template to an LLM into a reusable chain. This is where LangChain starts to shine.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Create components
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller"),
    ("user", "Write a 3-sentence story about {character} in {setting}")
])

# NEW: Output parser extracts just the text
output_parser = StrOutputParser()

# Connect them into a chain using the | operator (pipe)
chain = prompt | llm | output_parser

# Use the chain
result = chain.invoke({
    "character": "a robot",
    "setting": "ancient Rome"
})

print(result)  # Just the story text, no AIMessage wrapper
```

**What's happening here:**
1. `prompt` formats your input into messages
2. `llm` calls the AI model
3. `output_parser` extracts just the text string
4. The `|` operator chains them together

**The power of chains:** You can now reuse this entire workflow with different inputs:

```python
# Use the same chain for different stories
story1 = chain.invoke({"character": "a pirate", "setting": "Mars"})
story2 = chain.invoke({"character": "a dragon", "setting": "a library"})
story3 = chain.invoke({"character": "a time traveler", "setting": "the Stone Age"})
```

## Lesson 6: Sequential Chains - Multi-Step Processing

Sometimes you need multiple steps where the output of one becomes the input of the next.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Step 1: Generate a topic
topic_chain = (
    ChatPromptTemplate.from_messages([
        ("user", "Suggest one interesting topic related to {subject}")
    ])
    | llm
    | StrOutputParser()
)

# Step 2: Write about that topic
writing_chain = (
    ChatPromptTemplate.from_messages([
        ("user", "Write a short paragraph about: {topic}")
    ])
    | llm
    | StrOutputParser()
)

# Combine them
from operator import itemgetter

full_chain = (
    {"topic": topic_chain}  # First, get a topic
    | writing_chain          # Then, write about it
)

# Use it
result = full_chain.invoke({"subject": "space exploration"})
print(result)
```

**What's happening:**
1. First chain suggests a topic based on the subject
2. That topic becomes the input to the second chain
3. Second chain writes a paragraph about that topic
4. The entire flow runs automatically

This is sequential processing: output → input → output → input.

## Lesson 7: Simple Use Cases

Let's build three practical examples to cement these concepts.

**Use Case 1: Text Summarizer**

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

summarizer = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert at creating concise summaries"),
        ("user", "Summarize this text in {num_sentences} sentences:\n\n{text}")
    ])
    | llm
    | StrOutputParser()
)

# Use it
long_text = """
LangChain is a framework for developing applications powered by language models.
It enables applications that are context-aware and can reason about how to answer
based on provided information. The framework consists of several parts: LangChain
Libraries contain Python and JavaScript implementations, providing interfaces and
integrations for numerous components. LangChain Templates are a collection of easily
deployable reference architectures for a wide variety of tasks. LangServe is a library
for deploying LangChain chains as a REST API. LangSmith is a developer platform that
lets you debug, test, evaluate, and monitor chains built on any LLM framework.
"""

summary = summarizer.invoke({"text": long_text, "num_sentences": 2})
print("Summary:", summary)
```

**Use Case 2: Language Translator**

```python
translator = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator"),
        ("user", "Translate this text to {target_language}:\n\n{text}")
    ])
    | llm
    | StrOutputParser()
)

# Use it
result = translator.invoke({
    "text": "Hello, how are you today?",
    "target_language": "Spanish"
})
print("Translation:", result)
```

**Use Case 3: Email Generator**

```python
email_generator = (
    ChatPromptTemplate.from_messages([
        ("system", "You write professional business emails"),
        ("user", """Write a {tone} email about the following:
        
To: {recipient}
Subject: {subject}
Key points: {points}

Email:""")
    ])
    | llm
    | StrOutputParser()
)

# Use it
email = email_generator.invoke({
    "tone": "friendly but professional",
    "recipient": "the development team",
    "subject": "Upcoming code review meeting",
    "points": "Wednesday 2pm, Conference Room A, bring your laptops"
})
print(email)
```

## Lesson 8: Understanding LCEL (LangChain Expression Language)

You've been using LCEL already with the `|` operator. Let me explain what's really happening.

**LCEL is LangChain's way of composing chains using Python syntax.** The `|` operator (pipe) passes output from left to right, just like Unix pipes.

```python
# This LCEL chain:
chain = prompt | llm | output_parser

# Is equivalent to manually doing:
def manual_chain(input_data):
    formatted_prompt = prompt.format(**input_data)
    llm_response = llm.invoke(formatted_prompt)
    final_output = output_parser.parse(llm_response)
    return final_output
```

**Why LCEL matters:**
- Clean, readable syntax
- Automatic streaming support
- Parallel execution where possible
- Easy to modify and extend
- Type safety and validation

**LCEL Advanced Example - Parallel Execution:**

```python
from langchain.schema.runnable import RunnableParallel

# Create multiple chains
chain1 = (
    ChatPromptTemplate.from_template("What are the pros of {topic}?")
    | llm | StrOutputParser()
)

chain2 = (
    ChatPromptTemplate.from_template("What are the cons of {topic}?")
    | llm | StrOutputParser()
)

# Run them in parallel
parallel_chain = RunnableParallel(pros=chain1, cons=chain2)

# This runs both chains simultaneously and returns a dict
result = parallel_chain.invoke({"topic": "remote work"})
print("Pros:", result["pros"])
print("Cons:", result["cons"])
```

Both chains run at the same time, saving you time!

## Beginner Exercises

Now it's your turn to practice. Try building these:

**Exercise 1**: Create a chain that takes a person's name and favorite hobby, then generates a fun introduction for them (like an MC would do at an event).

**Exercise 2**: Build a "concept explainer" that takes a complex topic and the user's age, then explains it appropriately for that age level.

**Exercise 3**: Create a sequential chain that:
- Step 1: Generates 3 random topics
- Step 2: Picks the most interesting one
- Step 3: Writes a haiku about it

**Exercise 4**: Build a product description generator that takes a product name, features (comma-separated), and target audience, then creates a compelling product description.

Try these yourself before moving on. The best way to learn is by doing!

---

# Part 4: Intermediate Level - Building Real Applications {#part-4-intermediate}

## Lesson 9: Memory - Making Chatbots Remember

Up until now, our chains have been stateless. Each call is independent. But real chatbots need memory. Let me show you how to add it.

**Basic Conversation Memory:**

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Create memory
memory = ConversationBufferMemory()

# Create a conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # This shows you what's happening behind the scenes
)

# Have a conversation
print(conversation.predict(input="Hi, my name is Alice and I love Python"))
print(conversation.predict(input="What's my name?"))
print(conversation.predict(input="What programming language do I like?"))

# See the memory
print("\nConversation history:")
print(memory.buffer)
```

**What's happening:**
- `ConversationBufferMemory` stores the entire conversation history
- Each new message is automatically added to memory
- The LLM sees the full history with each call
- This allows the LLM to reference earlier parts of the conversation

**The problem with buffer memory:** It stores everything forever, which means it can get very long and expensive (more tokens = more cost).

**Solution: Sliding Window Memory:**

```python
from langchain.memory import ConversationBufferWindowMemory

# Only remember last 3 exchanges
memory = ConversationBufferWindowMemory(k=3)

conversation = ConversationChain(llm=llm, memory=memory)

# After many messages, only the last 3 are kept
for i in range(10):
    conversation.predict(input=f"Message number {i}")

print(memory.buffer)  # Only shows last 3 exchanges
```

**Solution: Summary Memory:**

```python
from langchain.memory import ConversationSummaryMemory

# This summarizes old messages to save tokens
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=memory)

# Have a long conversation
conversation.predict(input="Tell me about the history of Rome")
conversation.predict(input="What about Julius Caesar?")
conversation.predict(input="How did the empire fall?")

# Memory now contains a summary of the conversation
print(memory.buffer)
```

**Custom Memory for Specific Facts:**

```python
from langchain.memory import ConversationEntityMemory

# This tracks specific entities (people, places, things)
memory = ConversationEntityMemory(llm=llm)

conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="My friend Bob works at Google in San Francisco")
conversation.predict(input="Alice is Bob's manager and she loves coffee")

# Memory tracks entities
print(memory.entity_store.store)
# Shows: {"Bob": "works at Google in San Francisco", "Alice": "Bob's manager, loves coffee"}
```

## Lesson 10: Custom Memory in LCEL Chains

The conversation chain is convenient, but sometimes you want more control. Here's how to add memory to your custom LCEL chains.

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser

# Create a chat history object
history = ChatMessageHistory()

# Create a prompt template that includes history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),  # This is where history goes
    ("user", "{input}")
])

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create the chain
chain = prompt | llm | StrOutputParser()

# Function to chat with memory
def chat_with_memory(user_input):
    # Get messages from history
    messages = history.messages
    
    # Run the chain
    response = chain.invoke({
        "input": user_input,
        "history": messages
    })
    
    # Add to history
    history.add_user_message(user_input)
    history.add_ai_message(response)
    
    return response

# Use it
print(chat_with_memory("Hi, I'm learning LangChain"))
print(chat_with_memory("What am I learning?"))  # It remembers!
print(chat_with_memory("Can you quiz me on it?"))
```

**What makes this powerful:**
- You control exactly when and how memory is used
- You can inspect and modify the history
- You can save/load history from a database
- You can implement custom memory logic

## Lesson 11: Retrieval - Searching Through Documents

Now we enter one of LangChain's most powerful features: retrieval. This lets the LLM answer questions about documents that weren't in its training data.

**The basic flow:**
1. Load documents
2. Split them into chunks
3. Convert chunks to embeddings (vectors)
4. Store embeddings in a vector database
5. When user asks a question, find relevant chunks
6. Feed chunks + question to LLM

**Complete Example - Document Q&A:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Step 1: Load a document
# For this example, let's create a simple text document
with open("sample_doc.txt", "w") as f:
    f.write("""
    LangChain is a framework for developing applications powered by language models.
    It was created to help developers build context-aware reasoning applications.
    
    The key components are:
    - Models: Interfaces to LLMs
    - Prompts: Templates for inputs
    - Chains: Sequences of calls
    - Memory: Storing conversation state
    - Retrievers: Accessing data sources
    - Agents: Letting LLMs use tools
    
    LangChain supports many LLM providers including OpenAI, Anthropic, and open-source models.
    """)

loader = TextLoader("sample_doc.txt")
documents = loader.load()

# Step 2: Split into chunks
# Documents need to be split because LLMs have token limits
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Characters per chunk
    chunk_overlap=20  # Overlap between chunks (for context continuity)
)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")

# Step 3: Create embeddings and store in vector database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Save to disk
)

# Step 4: Create a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}  # Return top 2 most relevant chunks
)

# Step 5: Create a Q&A chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # Show which chunks were used
)

# Step 6: Ask questions!
question = "What are the key components of LangChain?"
result = qa_chain.invoke({"query": question})

print("Answer:", result["result"])
print("\nSource chunks used:")
for doc in result["source_documents"]:
    print("-", doc.page_content[:100])
```

**Understanding the components:**

**Text Splitter:** Why split documents? LLMs have token limits (e.g., 4096 tokens for GPT-3.5). Large documents won't fit. Plus, smaller chunks make search more precise.

**Embeddings:** These convert text to vectors (lists of numbers) that capture meaning. Similar concepts have similar vectors. This enables semantic search (search by meaning, not keywords).

**Vector Store:** A specialized database for storing and searching vectors. Chroma is simple and local. For production, use Pinecone, Weaviate, or Qdrant.

**Retriever:** The interface that searches the vector store. You ask it a question, it finds relevant chunks.

## Lesson 12: Advanced Retrieval Patterns

Basic retrieval is good, but let's make it better.

**Multi-Query Retrieval** (Generate multiple phrasings of the question):

```python
from langchain.retrievers import MultiQueryRetriever

# This generates multiple versions of the user's question
# Then retrieves for each version, increasing recall
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Now when you search, it generates variations like:
# "What is LangChain?" → ["What is LangChain?", "Explain LangChain", "LangChain definition"]
results = retriever.get_relevant_documents("What is LangChain?")
```

**Contextual Compression** (Only return the relevant parts of retrieved chunks):

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Create a compressor that extracts only relevant parts
compressor = LLMChainExtractor.from_llm(llm)

# Wrap your retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# Now retrieved chunks are compressed to only relevant sentences
results = compression_retriever.get_relevant_documents("What are LangChain's components?")
```

**Self-Query Retriever** (Parse user questions into filters):

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Define metadata about your documents
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source file",
        type="string"
    ),
    AttributeInfo(
        name="page",
        description="The page number",
        type="integer"
    )
]

document_content_description = "Technical documentation about LangChain"

# This can convert questions like "Show me pages about agents from the API docs"
# Into: query="agents" filter={"source": "api_docs"}
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info
)
```

## Lesson 13: Agents - Giving LLMs Tools

Agents are LLMs that can use tools. They reason about what tools to use, call them, and synthesize results. This is incredibly powerful.

**Basic Agent with Calculator:**

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, Tool
from langchain.agents import AgentExecutor
from langchain import hub

# Define a calculator tool
def calculator(expression):
    """Evaluates a mathematical expression"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

# Create the tool
calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Useful for doing math. Input should be a valid Python mathematical expression like '2 + 2' or '5 * 10'"
)

# Get a ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=[calculator_tool],
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[calculator_tool],
    verbose=True,  # Shows reasoning
    max_iterations=5
)

# Use it
result = agent_executor.invoke({
    "input": "What's 25 * 47 plus 189?"
})

print(result["output"])
```

**What happens when you run this:**

```
> Entering new AgentExecutor chain...
I need to calculate 25 * 47 first, then add 189

Action: Calculator
Action Input: 25 * 47
Observation: 1175
Thought: Now I need to add 189 to that result
Action: Calculator
Action Input: 1175 + 189
Observation: 1364
Thought: I now know the final answer
Final Answer: 1364
```

The agent is reasoning about what to do, using tools, and reaching conclusions!

**Agent with Multiple Tools:**

```python
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

# Tool 1: Calculator
calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="For math problems"
)

# Tool 2: Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="For looking up factual information"
)

# Create agent with both tools
agent = create_react_agent(llm=llm, tools=[calculator_tool, wikipedia_tool], prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=[calculator_tool, wikipedia_tool], verbose=True)

# Ask a question that requires both tools
result = agent_executor.invoke({
    "input": "Who was Albert Einstein and what year was he born? Calculate how old he would be in 2024."
})
```

The agent will:
1. Search Wikipedia for Einstein
2. Find his birth year (1879)
3. Use calculator to compute 2024 - 1879
4. Synthesize the answer

## Lesson 14: Custom Tools for Agents

You can give agents any tool you want. Let me show you how to create custom tools.

**Simple Function Tool:**

```python
from langchain.tools import Tool

def get_weather(location):
    """Mock weather API"""
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Rainy, 55°F",
        "Tokyo": "Cloudy, 68°F"
    }
    return weather_data.get(location, "Weather data not available")

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Get current weather for a location. Input should be a city name."
)

# Use it in an agent
agent_executor = AgentExecutor(
    agent=create_react_agent(llm, [weather_tool], prompt),
    tools=[weather_tool],
    verbose=True
)

result = agent_executor.invoke({"input": "What's the weather like in Tokyo?"})
```

**Tool with Structured Input** (for complex inputs):

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Define input schema
class EmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body content")

# Define the function
def send_email(to: str, subject: str, body: str) -> str:
    """Simulates sending an email"""
    return f"Email sent to {to} with subject '{subject}'"

# Create structured tool
email_tool = StructuredTool.from_function(
    func=send_email,
    name="SendEmail",
    description="Sends an email to a recipient",
    args_schema=EmailInput
)
```

**Tool that Calls an API:**

```python
import requests

def search_news(query):
    """Searches news via a mock API"""
    # In reality, you'd call a real API like NewsAPI
    # For demo purposes, we'll return mock data
    return f"Latest news about {query}: [Mock news articles would go here]"

news_tool = Tool(
    name="NewsSearch",
    func=search_news,
    description="Search for latest news articles on a topic"
)
```

## Lesson 15: Structured Output

Sometimes you want the LLM to return data in a specific format (JSON, Pydantic object, etc.) instead of free text.

**Using Output Parsers:**

```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define your desired output structure
class Recipe(BaseModel):
    name: str = Field(description="Name of the dish")
    ingredients: list[str] = Field(description="List of ingredients")
    instructions: str = Field(description="Cooking instructions")
    prep_time: int = Field(description="Preparation time in minutes")

# Create parser
parser = PydanticOutputParser(pydantic_object=Recipe)

# Create prompt with format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chef. Always respond with valid JSON."),
    ("user", "{query}\n\n{format_instructions}")
])

# Create chain
chain = (
    prompt 
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | parser  # This parses output into Recipe object
)

# Use it
result = chain.invoke({
    "query": "Give me a recipe for chocolate chip cookies",
    "format_instructions": parser.get_format_instructions()
})

print(result.name)
print(result.ingredients)
print(result.prep_time)
```

**JSON Mode** (newer models support native JSON):

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    model_kwargs={"response_format": {"type": "json_object"}}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You respond only in valid JSON"),
    ("user", "List 3 programming languages with their use cases")
])

chain = prompt | llm
result = chain.invoke({})
print(result.content)  # Guaranteed to be valid JSON
```

## Lesson 16: RAG (Retrieval-Augmented Generation) - Putting It All Together

RAG combines retrieval with generation. The LLM retrieves relevant information and uses it to answer questions. This is one of the most important patterns.

**Complete RAG System:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate

# Step 1: Load multiple documents
loader = DirectoryLoader('./documents', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Step 2: Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag_db"
)

# Step 4: Create custom prompt for RAG
template = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Step 5: Create RAG chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Use it
result = qa_chain.invoke({"query": "What are the main topics discussed?"})
print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata}")
```

**RAG with Chat History** (conversational RAG):

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create conversational RAG chain
conversational_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Have a conversation about your documents
print(conversational_qa({"question": "What is this document about?"}))
print(conversational_qa({"question": "Can you elaborate on the first point?"}))
print(conversational_qa({"question": "How does that relate to what you said earlier?"}))
```

Now the system remembers the conversation AND retrieves from documents!

## Intermediate Exercises

Try building these more complex applications:

**Exercise 1**: Build a personal journal analyzer. Store journal entries, let users ask questions like "What was I worried about last month?" or "What made me happy this year?"

**Exercise 2**: Create an agent that can search Wikipedia AND do calculations. Ask it questions like "How old is the oldest building in Rome?" (requires searching + calculating current year - building year)

**Exercise 3**: Build a code documentation Q&A system. Load Python code or documentation, let users ask how to use specific functions.

**Exercise 4**: Create a customer support bot with memory and knowledge base. It should remember the customer's previous questions and search a FAQ document.

---

# Part 5: Advanced Level - Production Systems {#part-5-advanced}

## Lesson 17: LangGraph - Stateful, Controlled Workflows

LangGraph is LangChain's solution for building complex, stateful agent workflows. It gives you fine-grained control over agent behavior.

**Why LangGraph?** Standard agents make all decisions autonomously. LangGraph lets you define the control flow explicitly.

**Basic LangGraph Example:**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class State(TypedDict):
    messages: Annotated[list, operator.add]
    current_step: str
    result: str

# Define nodes (steps in the workflow)
def research_step(state):
    """First step: research"""
    print("Researching...")
    state["current_step"] = "research"
    state["messages"].append("Research complete")
    return state

def analysis_step(state):
    """Second step: analyze"""
    print("Analyzing...")
    state["current_step"] = "analysis"
    state["messages"].append("Analysis complete")
    return state

def report_step(state):
    """Final step: report"""
    print("Generating report...")
    state["result"] = "Final report: " + ", ".join(state["messages"])
    return state

# Build graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("research", research_step)
workflow.add_node("analysis", analysis_step)
workflow.add_node("report", report_step)

# Define edges (flow)
workflow.set_entry_point("research")
workflow.add_edge("research", "analysis")
workflow.add_edge("analysis", "report")
workflow.add_edge("report", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({
    "messages": [],
    "current_step": "",
    "result": ""
})

print(result["result"])
```

**Conditional Branching:**

```python
def should_continue(state):
    """Decide whether to continue or end"""
    if len(state["messages"]) > 5:
        return "end"
    return "continue"

workflow = StateGraph(State)
workflow.add_node("process", process_step)
workflow.add_node("continue_process", continue_step)

workflow.set_entry_point("process")

# Conditional edge
workflow.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "continue_process",
        "end": END
    }
)
```

## Lesson 18: Multi-Agent Systems

Sometimes you want multiple specialized agents working together. One common pattern is Supervisor + Workers.

**Supervisor-Worker Pattern:**

```python
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub

# Worker 1: Research Agent
research_prompt = hub.pull("hwchase17/react")
research_llm = ChatOpenAI(model="gpt-3.5-turbo")

def research_web(query):
    """Mock web research"""
    return f"Research results for {query}: [Mock data]"

research_tools = [Tool(
    name="WebSearch",
    func=research_web,
    description="Search the web for information"
)]

research_agent = create_react_agent(research_llm, research_tools, research_prompt)
research_executor = AgentExecutor(agent=research_agent, tools=research_tools)

# Worker 2: Analysis Agent
def analyze_data(data):
    """Mock data analysis"""
    return f"Analysis of {data}: [Mock analysis]"

analysis_tools = [Tool(
    name="Analyzer",
    func=analyze_data,
    description="Analyze data and provide insights"
)]

analysis_agent = create_react_agent(research_llm, analysis_tools, research_prompt)
analysis_executor = AgentExecutor(agent=analysis_agent, tools=analysis_tools)

# Supervisor: Coordinates workers
def supervisor(task):
    """Supervisor decides which agent to use"""
    if "research" in task.lower() or "find" in task.lower():
        return research_executor.invoke({"input": task})
    elif "analyze" in task.lower():
        return analysis_executor.invoke({"input": task})
    else:
        return {"output": "Task not understood"}

# Use the system
result = supervisor("Research the latest trends in AI")
print(result)

result = supervisor("Analyze this dataset")
print(result)
```

**Collaborative Multi-Agent System:**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MultiAgentState(TypedDict):
    task: str
    research_results: str
    analysis_results: str
    final_report: str

# Agent 1: Researcher
def researcher_node(state):
    # Call research agent
    result = research_executor.invoke({"input": state["task"]})
    state["research_results"] = result["output"]
    return state

# Agent 2: Analyst
def analyst_node(state):
    # Analyze research results
    task = f"Analyze: {state['research_results']}"
    result = analysis_executor.invoke({"input": task})
    state["analysis_results"] = result["output"]
    return state

# Agent 3: Writer
def writer_node(state):
    # Synthesize into report
    prompt = f"Write a report based on:\nResearch: {state['research_results']}\nAnalysis: {state['analysis_results']}"
    # Call LLM to generate report
    llm = ChatOpenAI(model="gpt-4")
    report = llm.invoke(prompt)
    state["final_report"] = report.content
    return state

# Build workflow
workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# Run the multi-agent system
result = app.invoke({
    "task": "Research and analyze recent developments in quantum computing",
    "research_results": "",
    "analysis_results": "",
    "final_report": ""
})

print(result["final_report"])
```

## Lesson 19: Streaming Responses

For better UX, you want to stream LLM responses token-by-token rather than waiting for the complete response.

**Streaming with LCEL:**

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

prompt = ChatPromptTemplate.from_messages([
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# Stream the response
for chunk in chain.stream({"question": "Write a long story about a space adventure"}):
    print(chunk, end="", flush=True)
```

**Streaming with AgentExecutor:**

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Now responses stream in real-time
result = agent_executor.invoke({"input": "Tell me a story"})
```

## Lesson 20: Async Operations

For high-throughput applications, use async operations to handle multiple requests concurrently.

**Async LCEL:**

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("Tell me a fact about {topic}")
chain = prompt | llm

async def process_multiple_topics():
    topics = ["Python", "space", "history", "biology", "physics"]
    
    # Process all topics concurrently
    tasks = [chain.ainvoke({"topic": topic}) for topic in topics]
    results = await asyncio.gather(*tasks)
    
    for topic, result in zip(topics, results):
        print(f"{topic}: {result.content}\n")

# Run
asyncio.run(process_multiple_topics())
```

This processes all 5 topics simultaneously instead of one at a time!

## Lesson 21: Custom LLM Integration

You can integrate any LLM, including local models, with LangChain.

**Integrating a Local Model:**

```python
from langchain.llms.base import LLM
from typing import Optional, List, Any

class CustomLocalLLM(LLM):
    """Custom LLM wrapper for a local model"""
    
    model_path: str
    
    @property
    def _llm_type(self) -> str:
        return "custom_local"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Call the local model"""
        # Here you would call your actual local model
        # For example, using llama-cpp-python or transformers
        
        # Mock implementation
        return f"Response from local model for: {prompt}"

# Use it like any other LLM
local_llm = CustomLocalLLM(model_path="/path/to/model")
response = local_llm.invoke("Hello, how are you?")
print(response)
```

**Using Ollama (Local LLM):**

```python
from langchain.llms import Ollama

# Ollama must be running locally
llm = Ollama(model="llama2")

response = llm.invoke("Explain quantum computing")
print(response)
```

## Lesson 22: Advanced RAG Techniques

Let's explore sophisticated RAG patterns for production systems.

**Hybrid Search (Keyword + Semantic):**

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Keyword-based retriever
bm25_retriever = BM25Retriever.from_documents(documents)

# Semantic retriever
embedding_retriever = vectorstore.as_retriever()

# Combine both with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, embedding_retriever],
    weights=[0.4, 0.6]  # 40% keyword, 60% semantic
)

# This retrieves using both methods and merges results
results = ensemble_retriever.get_relevant_documents("What is LangChain?")
```

**Re-ranking Retrieved Documents:**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# First, retrieve candidates
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Then, re-rank to get top 3 most relevant
compressor = CohereRerank(top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

results = compression_retriever.get_relevant_documents("question")
```

**Parent Document Retriever** (Retrieve small chunks, return larger context):

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store for parent documents
store = InMemoryStore()

# Small chunks for retrieval (precise matching)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# Large chunks for context (better LLM comprehension)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

retriever.add_documents(documents)

# Retrieves small chunks but returns large context
results = retriever.get_relevant_documents("question")
```

**Query Expansion:**

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Generate multiple versions of the query
expansion_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 3 different versions of this question to search for information:

Original: {question}

Version 1:
Version 2:
Version 3:"""
)

expansion_chain = LLMChain(llm=llm, prompt=expansion_prompt)

def expanded_retrieval(question):
    # Generate variations
    variations = expansion_chain.invoke({"question": question})
    
    # Parse variations (simplified)
    questions = [question] + variations["text"].split("\n")
    
    # Retrieve for each
    all_docs = []
    for q in questions:
        docs = vectorstore.similarity_search(q, k=3)
        all_docs.extend(docs)
    
    # Deduplicate
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
    return unique_docs[:5]
```

## Lesson 23: Evaluation and Testing

Production systems need evaluation. Here's how to test your LangChain applications.

**Testing RAG Systems:**

```python
from langchain.evaluation import load_evaluator

# Create an evaluator
evaluator = load_evaluator("qa")

# Test your QA chain
question = "What is LangChain?"
prediction = qa_chain.invoke({"query": question})
reference = "LangChain is a framework for building LLM applications"

# Evaluate
result = evaluator.evaluate_strings(
    prediction=prediction["result"],
    reference=reference,
    input=question
)

print(f"Score: {result['score']}")
```

**Custom Evaluation:**

```python
from langchain.evaluation import StringEvaluator

class CustomEvaluator(StringEvaluator):
    """Custom evaluator for specific criteria"""
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs
    ) -> dict:
        # Your custom evaluation logic
        score = 0.0
        
        # Check if answer is factual
        if self.is_factual(prediction):
            score += 0.5
        
        # Check if answer is complete
        if self.is_complete(prediction, input):
            score += 0.5
        
        return {"score": score}
    
    def is_factual(self, text):
        # Use an LLM to check factuality
        return True  # Simplified
    
    def is_complete(self, answer, question):
        # Check if answer addresses the question
        return True  # Simplified

evaluator = CustomEvaluator()
result = evaluator.evaluate_strings(prediction="...", input="...")
```

## Lesson 24: Error Handling and Retries

Production systems need robust error handling.

**Retry Logic:**

```python
from langchain.llms import OpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from tenacity import retry, stop_after_attempt, wait_exponential

class RetryLLM(OpenAI):
    """LLM with automatic retries"""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call with retries"""
        try:
            return super()._call(prompt, stop=stop, **kwargs)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            raise

llm = RetryLLM()
# Now failures will retry automatically
```

**Fallback LLMs:**

```python
from langchain.llms import OpenAI, Anthropic

primary_llm = OpenAI(model="gpt-4")
fallback_llm = Anthropic(model="claude-2")

def call_with_fallback(prompt):
    """Try primary, fall back to secondary"""
    try:
        return primary_llm.invoke(prompt)
    except Exception as e:
        print(f"Primary failed: {e}. Using fallback...")
        return fallback_llm.invoke(prompt)

response = call_with_fallback("Hello")
```

## Lesson 25: Monitoring and Logging

Track your LLM calls for debugging and cost management.

**Using LangSmith:**

```python
import os

# Enable LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# Now all chain runs are automatically logged to LangSmith
chain = prompt | llm
result = chain.invoke({"input": "test"})

# View traces at smith.langchain.com
```

**Custom Logging:**

```python
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Log to file
file_handler = FileCallbackHandler("langchain.log")

# Multiple callbacks
llm = ChatOpenAI(
    callbacks=[
        file_handler,
        StreamingStdOutCallbackHandler()
    ]
)

# All calls are logged to file AND streamed to stdout
```

## Advanced Exercises

These are challenging real-world projects:

**Exercise 1**: Build a complete RAG system with hybrid search, re-ranking, and query expansion. Evaluate it on a test dataset.

**Exercise 2**: Create a multi-agent system where one agent researches, another analyzes, and a third writes reports. Use LangGraph for control flow.

**Exercise 3**: Build an agent that can query SQL databases, search the web, and do calculations. Test it with complex multi-step questions.

**Exercise 4**: Implement a production-ready chatbot with streaming, error handling, logging, and cost tracking.

---

# Part 6: Complete Projects {#part-6-projects}

## Project 1: Document Q&A Chatbot with Memory

**Goal**: Build a chatbot that answers questions about uploaded documents and remembers conversation history.

**Features**:
- Upload multiple PDFs/text files
- Split and embed documents
- Conversational interface with memory
- Show source citations

**Complete Implementation**:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

class DocumentChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.conversation_chain = None
        
    def load_documents(self, directory):
        """Load all documents from a directory"""
        # Load PDFs
        pdf_loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = pdf_loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./doc_chatbot_db"
        )
        
        # Create conversation chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )
        
        print(f"Loaded {len(chunks)} chunks from {len(documents)} documents")
    
    def ask(self, question):
        """Ask a question"""
        if not self.conversation_chain:
            return "Please load documents first!"
        
        result = self.conversation_chain({"question": question})
        
        # Format response with sources
        answer = result["answer"]
        sources = result["source_documents"]
        
        response = f"Answer: {answer}\n\nSources:\n"
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            response += f"{i}. {source} (Page {page})\n"
        
        return response

# Usage
chatbot = DocumentChatbot()
chatbot.load_documents("./documents")

# Have a conversation
print(chatbot.ask("What is this document about?"))
print(chatbot.ask("Can you give me more details about the first point?"))
print(chatbot.ask("How does that relate to what you said earlier?"))
```

## Project 2: Customer Support Assistant

**Goal**: Build an intelligent customer support system with tools, memory, and a knowledge base.

**Features**:
- Search knowledge base (FAQs)
- Look up order status (mock DB)
- Send emails (mock)
- Remember customer context

**Complete Implementation**:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import hub

class CustomerSupportAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.setup_knowledge_base()
        self.setup_tools()
        self.setup_agent()
    
    def setup_knowledge_base(self):
        """Load FAQs into vector store"""
        faqs = [
            "Q: How do I reset my password? A: Click 'Forgot Password' on the login page",
            "Q: What's your return policy? A: 30-day full refund, no questions asked",
            "Q: How long does shipping take? A: 3-5 business days for standard shipping",
            # Add more FAQs
        ]
        
        from langchain.schema import Document
        docs = [Document(page_content=faq) for faq in faqs]
        
        self.knowledge_base = Chroma.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings()
        )
    
    def setup_tools(self):
        """Create tools for the agent"""
        
        def search_kb(query):
            """Search knowledge base"""
            results = self.knowledge_base.similarity_search(query, k=2)
            return "\n".join([doc.page_content for doc in results])
        
        def get_order_status(order_id):
            """Mock order lookup"""
            orders = {
                "12345": "Shipped - Arriving Oct 5",
                "67890": "Processing - Will ship Oct 3"
            }
            return orders.get(order_id, "Order not found")
        
        def send_email(to, subject, body):
            """Mock email sending"""
            return f"Email sent to {to}: {subject}"
        
        self.tools = [
            Tool(
                name="SearchKnowledgeBase",
                func=search_kb,
                description="Search FAQs and documentation"
            ),
            Tool(
                name="GetOrderStatus",
                func=get_order_status,
                description="Look up order status by order ID"
            ),
            Tool(
                name="SendEmail",
                func=send_email,
                description="Send an email to a customer"
            )
        ]
    
    def setup_agent(self):
        """Create agent with memory"""
        prompt = hub.pull("hwchase17/react")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history"
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            max_iterations=5
        )
    
    def chat(self, message):
        """Handle customer message"""
        result = self.agent_executor.invoke({"input": message})
        return result["output"]

# Usage
support_agent = CustomerSupportAgent()

print(support_agent.chat("Hi, what's your return policy?"))
print(support_agent.chat("Great! Can you check order 12345?"))
print(support_agent.chat("When will it arrive?"))
```

## Project 3: Multi-Agent Research Assistant

**Goal**: Build a research system with specialized agents for different tasks.

**Features**:
- Researcher agent (web search)
- Analyst agent (data analysis)
- Writer agent (report generation)
- Coordinator agent (manages workflow)

**Complete Implementation**:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import operator

class ResearchState(TypedDict):
    topic: str
    research_query: str
    research_results: Annotated[list, operator.add]
    analysis: str
    final_report: str
    steps_completed: Annotated[list, operator.add]

class ResearchAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.setup_workflow()
    
    def planner_node(self, state):
        """Plan the research"""
        prompt = ChatPromptTemplate.from_template(
            "Given the topic '{topic}', create a focused research query:"
        )
        chain = prompt | self.llm | StrOutputParser()
        query = chain.invoke({"topic": state["topic"]})
        
        state["research_query"] = query
        state["steps_completed"].append("planning")
        return state
    
    def researcher_node(self, state):
        """Conduct research (mock)"""
        # In reality, use web search tool
        prompt = ChatPromptTemplate.from_template(
            "Provide 3 key facts about: {query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        facts = chain.invoke({"query": state["research_query"]})
        
        state["research_results"].append(facts)
        state["steps_completed"].append("research")
        return state
    
    def analyst_node(self, state):
        """Analyze findings"""
        prompt = ChatPromptTemplate.from_template(
            """Analyze these research findings and provide insights:
            
            {findings}
            
            Provide a structured analysis:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        
        findings = "\n".join(state["research_results"])
        analysis = chain.invoke({"findings": findings})
        
        state["analysis"] = analysis
        state["steps_completed"].append("analysis")
        return state
    
    def writer_node(self, state):
        """Generate final report"""
        prompt = ChatPromptTemplate.from_template(
            """Create a comprehensive report on: {topic}
            
            Research: {research}
            Analysis: {analysis}
            
            Write a professional report:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        
        report = chain.invoke({
            "topic": state["topic"],
            "research": "\n".join(state["research_results"]),
            "analysis": state["analysis"]
        })
        
        state["final_report"] = report
        state["steps_completed"].append("writing")
        return state
    
    def setup_workflow(self):
        """Build the workflow graph"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("writer", self.writer_node)
        
        # Define flow
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "writer")
        workflow.add_edge("writer", END)
        
        self.app = workflow.compile()
    
    def research(self, topic):
        """Run research on a topic"""
        result = self.app.invoke({
            "topic": topic,
            "research_query": "",
            "research_results": [],
            "analysis": "",
            "final_report": "",
            "steps_completed": []
        })
        return result["final_report"]

# Usage
assistant = ResearchAssistant()
report = assistant.research("The impact of quantum computing on cryptography")
print(report)
```

## Project 4: AI Code Interpreter

**Goal**: Build an agent that can write and execute Python code safely.

**Features**:
- Write Python code based on user requests
- Execute code in a sandboxed environment
- Handle errors and retry
- Return results to user

**Complete Implementation**:

```python
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
import subprocess
import tempfile
import os

class CodeInterpreter:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.setup_tools()
        self.setup_agent()
    
    def execute_python(self, code):
        """Execute Python code safely"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return f"Success!\nOutput:\n{result.stdout}"
            else:
                return f"Error:\n{result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def setup_tools(self):
        """Create tools"""
        self.tools = [
            Tool(
                name="PythonExecutor",
                func=self.execute_python,
                description="""Execute Python code. Input should be valid Python code.
                Use this to perform calculations, data analysis, or any Python operations."""
            )
        ]
    
    def setup_agent(self):
        """Create agent"""
        prompt = hub.pull("hwchase17/react")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )
    
    def run(self, task):
        """Execute a task"""
        result = self.agent_executor.invoke({"input": task})
        return result["output"]

# Usage
interpreter = CodeInterpreter()

print(interpreter.run("Calculate the first 10 Fibonacci numbers"))
print(interpreter.run("Create a list of prime numbers under 100"))
print(interpreter.run("Generate a random 10x10 matrix and find its determinant"))
```

## Project 5: Enterprise RAG Pipeline

**Goal**: Build a production-ready RAG system with all the bells and whistles.

**Features**:
- Hybrid search (keyword + semantic)
- Re-ranking
- Query expansion
- Streaming responses
- Error handling
- Logging and monitoring
- Cost tracking

**Complete Implementation**: This would be very long, so I'll provide the architecture:

```python
class EnterpriseRAG:
    """Production-ready RAG system"""
    
    def __init__(self):
        self.setup_components()
        self.setup_monitoring()
    
    def setup_components(self):
        # Vector store with persistence
        # Multiple embedding models
        # Hybrid retriever
        # Re-ranker
        # LLM with fallbacks
        # Streaming handler
        pass
    
    def setup_monitoring(self):
        # LangSmith integration
        # Custom logging
        # Cost tracking
        # Performance metrics
        pass
    
    def ingest_documents(self, paths):
        # Load documents
        # OCR for images
        # Extract metadata
        # Split intelligently
        # Generate embeddings
        # Store with metadata
        pass
    
    def query(self, question, stream=True):
        # Expand query
        # Retrieve with hybrid search
        # Re-rank results
        # Generate answer
        # Stream response
        # Log everything
        pass
```

---

# Part 7: Best Practices & Optimization {#part-7-best-practices}

## Best Practices Summary

**1. Prompt Engineering**
- Be specific and clear in your instructions
- Provide examples (few-shot learning)
- Use system messages to set behavior
- Test prompts iteratively
- Version control your prompts

**2. Cost Optimization**
- Use cheaper models (GPT-3.5) for simple tasks
- Cache repeated queries
- Implement streaming to reduce wasted tokens
- Trim conversation history
- Use summary memory instead of full history

**3. Performance**
- Use async for concurrent operations
- Batch similar requests
- Cache embeddings
- Use smaller embedding models for large datasets
- Implement timeouts and retries

**4. Security**
- Never hardcode API keys
- Use environment variables
- Validate user inputs
- Sanitize outputs
- Rate limit API calls
- Implement proper error handling

**5. Monitoring**
- Use LangSmith for tracing
- Log all LLM calls
- Track token usage and costs
- Monitor latency
- Set up alerts for failures

## Common Mistakes and Solutions

**Mistake 1: Not handling token limits**
Solution: Implement chunking and summary memory

**Mistake 2: Ignoring errors**
Solution: Add retry logic and fallback LLMs

**Mistake 3: Poor retrieval**
Solution: Use hybrid search, re-ranking, and query expansion

**Mistake 4: No evaluation**
Solution: Create test sets and evaluate regularly

**Mistake 5: Blocking operations**
Solution: Use async for I/O operations

## Debugging Tips

**Problem: Chain not working**
- Set `verbose=True` to see what's happening
- Check inputs/outputs at each step
- Validate prompt templates
- Test components individually

**Problem: Poor retrieval results**
- Check chunk size and overlap
- Try different embedding models
- Inspect actual retrieved chunks
- Experiment with search parameters

**Problem: High costs**
- Monitor token usage per call
- Use cheaper models where possible
- Implement caching
- Trim conversation history

**Problem: Slow performance**
- Profile your code
- Use async operations
- Reduce chunk size
- Optimize retrieval parameters

## Production Readiness Checklist

- [ ] Environment variables for API keys
- [ ] Error handling and retries
- [ ] Logging and monitoring
- [ ] Rate limiting
- [ ] Input validation
- [ ] Output sanitization
- [ ] Cost tracking
- [ ] Performance metrics
- [ ] Evaluation pipeline
- [ ] Documentation
- [ ] Unit tests
- [ ] Integration tests
- [ ] Deployment configuration
- [ ] Backup and recovery

## Next Steps and Resources

**Congratulations!** You've completed this comprehensive LangChain guide. Here's what to do next:

1. **Build Your Own Project**: Apply what you've learned to a real problem
2. **Join the Community**: LangChain Discord, GitHub discussions
3. **Read the Docs**: Official LangChain documentation
4. **Explore Integrations**: Try different LLMs, vector stores, tools
5. **Contribute**: Open source contributions are welcome
6. **Stay Updated**: LangChain evolves rapidly

**Resources:**
- LangChain Docs: https://python.langchain.com
- LangChain GitHub: https://github.com/langchain-ai/langchain
- LangSmith: https://smith.langchain.com
- LangChain Blog: https://blog.langchain.dev

**Remember**: The best way to master LangChain is by building. Start with simple projects and gradually increase complexity. Don't be afraid to experiment and make mistakes. That's how you learn!

Good luck on your LangChain journey! 🚀
