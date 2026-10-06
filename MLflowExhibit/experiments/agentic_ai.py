import mlflow
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    ToolCallCorrectness,
    ToolCallEfficiency,
)

from mlflow.genai import scorer

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Agentic_AI")

mlflow.langchain.autolog()

@tool
def get_account_balance(account_id: str) -> str:
    """Get the current balance of a bank account."""

    # Fake database result for learning
    balances = {
        "A100": 2500.0,
        "A200": 7800.0,
    }

    balance = balances.get(account_id)

    if balance is None:
        return "Account not found."

    return f"The account balance is ${balance:.2f}"


@tool
def get_recent_transactions(account_id: str) -> str:
    """Get recent transactions for a bank account."""

    # Fake database result
    transactions = {
        "A100": [
            "Woolworths: -$85",
            "Salary: +$3200",
            "Netflix: -$25",
        ]
    }

    result = transactions.get(account_id)

    if result is None:
        return "No transactions found."

    return "\n".join(result)


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

tools = [
    get_account_balance,
    get_recent_transactions,
]

agent = create_agent(
    model=llm,
    tools=tools,
)

def banking_agent(question: str) -> str:

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    return result["messages"][-1].content

question = "What is the balance of account A100?"

answer = banking_agent(question)

print("Question:", question)
print("Answer:", answer)


question = mlflow.genai.load_prompt('prompts:/Trans/3')
answer = banking_agent(question.template)

print("Question:", question)
print("Answer:", answer)


question = mlflow.genai.load_prompt('prompts:/Trans/4')
answer = banking_agent(question.template)

print("Question:", question)
print("Answer:", answer)

eval_data = [
    {
        "inputs": {
            "question": "What is the balance of account A100?"
        },
        "expectations": {
            "expected_response":
                "The balance of account A100 is $2500."
        },
    },

    {
        "inputs": {
            "question":
                "Show me the recent transactions for account A100."
        },
        "expectations": {
            "expected_response":
                "Recent transactions include Woolworths -$85, "
                "Salary +$3200, and Netflix -$25."
        },
    },
]

@scorer
def is_concise(outputs: str) -> bool:
    """Is the response concise?"""
    return len(outputs.split()) < 5


results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=banking_agent,
    scorers=[
        Correctness(),
        RelevanceToQuery(),
        ToolCallCorrectness(),
        ToolCallEfficiency(),
        is_concise,
    ],
)


print("Evaluation completed.")