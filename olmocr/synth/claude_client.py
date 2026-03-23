import asyncio
import re

from anthropic import AsyncAnthropic

DEFAULT_MODEL_NAME = "claude-sonnet-4-6"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 5


def _is_overloaded_error(error: Exception) -> bool:
    """Return True if the error indicates the Claude service is overloaded."""
    return "Overloaded" in str(error)


async def call_claude(
    client: AsyncAnthropic,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_BACKOFF_SECONDS,
    **kwargs,
):
    """Call Claude with exponential backoff when the service is overloaded."""
    delay = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            return await client.messages.create(**kwargs)
        except Exception as error:
            if not _is_overloaded_error(error) or attempt == max_retries:
                raise

            await asyncio.sleep(delay)
            delay *= 2


async def claude_stream(
    client: AsyncAnthropic,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_BACKOFF_SECONDS,
    **kwargs,
):
    """Call Claude streaming endpoint and return final message with overload retries."""
    delay = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            async with client.messages.stream(**kwargs) as stream:
                async for _ in stream:
                    pass
                return await stream.get_final_message()
        except Exception as error:
            if not _is_overloaded_error(error) or attempt == max_retries:
                raise

            await asyncio.sleep(delay)
            delay *= 2


def extract_code_block(initial_response):
    # Use regex to find the last instance of a code block
    # First try to find HTML specific code blocks
    html_blocks = re.findall(r"```html\n(.*?)```", initial_response, re.DOTALL)

    # If HTML blocks found, return the last one
    if html_blocks:
        return html_blocks[-1].strip()

    # Otherwise, try to find any code blocks
    code_blocks = re.findall(r"```\n(.*?)```", initial_response, re.DOTALL)

    # If code blocks found, return the last one
    if code_blocks:
        return code_blocks[-1].strip()

    # If no code blocks found with newlines after backticks, try without newlines
    html_blocks_no_newline = re.findall(r"```html(.*?)```", initial_response, re.DOTALL)
    if html_blocks_no_newline:
        return html_blocks_no_newline[-1].strip()

    code_blocks_no_newline = re.findall(r"```(.*?)```", initial_response, re.DOTALL)
    if code_blocks_no_newline:
        return code_blocks_no_newline[-1].strip()

    # Return empty string if no code blocks found
    return None
