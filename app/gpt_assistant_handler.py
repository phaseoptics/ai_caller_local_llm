import os
import asyncio
import logging
import warnings
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Suppress OpenAI SDK and httpx info logs unless explicitly enabled
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

async def init_conversation(thread_id: Optional[str] = None, logging_enabled: bool = False) -> str:
    logger = logging.getLogger("gpt_handler")
    logger.setLevel(logging.INFO if logging_enabled else logging.CRITICAL)

    if thread_id:
        logger.info(f"Resuming existing thread: {thread_id}")
        return thread_id
    thread = await client.beta.threads.create()
    logger.info(f"Created new thread: {thread.id}")
    return thread.id

async def send_user_message(thread_id: str, user_input: str, logging_enabled: bool = False) -> str:
    logger = logging.getLogger("gpt_handler")
    logger.setLevel(logging.INFO if logging_enabled else logging.CRITICAL)

    logger.info(f"Sending message to thread {thread_id}: {user_input}")

    await client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=[{"type": "text", "text": user_input}]
    )

    run = await client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )
    logger.info(f"Started run: {run.id}")

    while True:
        run = await client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status == "completed":
            logger.info("Run completed.")
            break
        elif run.status in ["failed", "cancelled"]:
            logger.error(f"Run failed with status: {run.status}")
            raise RuntimeError(f"Run failed: {run.status}")
        await asyncio.sleep(0.1)

    messages = await client.beta.threads.messages.list(thread_id=thread_id)
    sorted_messages = sorted(messages.data, key=lambda m: m.created_at, reverse=True)

    for message in sorted_messages:
        if message.role == "assistant":
            for part in message.content:
                if part.type == "text":
                    reply = part.text.value.strip()
                    logger.info(f"Assistant reply: {reply}")
                    return reply

    logger.warning("No assistant reply found.")
    return "[No assistant reply found]"


# Expanded async test block to verify memory persistence
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.CRITICAL, format='[%(levelname)s] %(message)s')
        logging.info("Starting GPT assistant memory test...")

        tid = await init_conversation(logging_enabled=False)

        input1 = "Please remember that my favorite number is 8675309."
        reply1 = await send_user_message(tid, input1, logging_enabled=False)
        print("User:", input1)
        print("Assistant:", reply1)

        input2 = "Who painted the Mona Lisa?"
        reply2 = await send_user_message(tid, input2, logging_enabled=False)
        print("User:", input2)
        print("Assistant:", reply2)

        input3 = "What is my favorite number? Do you know why"
        reply3 = await send_user_message(tid, input3, logging_enabled=False)
        print("User:", input3)
        print("Assistant:", reply3)

    asyncio.run(main())