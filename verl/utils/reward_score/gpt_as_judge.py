import asyncio
import json
import os
import httpx
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai import APIError, APITimeoutError, APIConnectionError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio
import logging
import time
from .resilience import EndpointManager, CircuitBreakerConfig

logger = logging.getLogger(__name__)


def get_compare_messages(question, response, answer):
    prompt = f"""
Your task is to determine whether the user's answer is correct based on the provided questions and standard answers (for example, if the user expresses a similar meaning to the standard answer, or another interpretation of the standard answer, it is considered correct.)

Note(very important!):
1.If the standard answer is an interval, and the user's answer is only one value, it is considered wrong. If the standard answer has only one option, and the user's answer has multiple options, it is also considered wrong.
2. If the user's answer has no unit, but the value is consistent with the standard answer, it is also considered correct, such as 100 and 100m, 25t and 25,they are considered to be the same.
3. If the answer is an equation and the answer is just a value, it is OK as long as the value is consistent with the value after the equation, such as area=pi/25\xa0^2 is same as pi/25.

The question is: {question}

The standard answer: {answer}

The user's answer: {response}

Please strictly follow the following format for output(0 represents correct, 1 represents incorrect):
<think>{{your concise think step}}</think>
<judge>{{0/1}}</judge>

for example:
<think>The standard answer is right, and the user's answer is right frontal lobe, they express the same meaning, so it is correct.</think>
<judge>0</judge>

<think>The standard answer is 0.5, and the user's answer is \\frac{{1}}{{2}}. The numerical calculations of the two are consistent, so it is correct.</think>
<judge>0</judge>

<think>The standard answer is 0.5, and the user's answer is \\frac{{1}}{{3}}. The numerical calculations of the two are inconsistent, so it is incorrect.</think>
<judge>1</judge>

<think>The standard answer is 2t, and the user's answer is 2. The value before the unit is the same, so it is correct.</think>
<judge>0</judge>
    """
    messages = [{"role": "user", "content": prompt}]
    return messages


class fake_response:
    def __init__(self, usage):
        self.usage = usage


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        logger.warning(
            f"Retrying API call. Attempt #{retry_state.attempt_number}, "
            f"Last exception: {retry_state.outcome.exception()}"
        )


async def deal_tasks(tasks, max_concurrent_tasks=256):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_task(task):
        async with semaphore:
            return await task  # 注意这里是调用task()

    # 创建未执行的协程列表
    sem_tasks = [sem_task(task) for task in tasks]

    # 使用tqdm_asyncio.gather来调度任务并显示进度
    print("Calling model to verify answers...")
    for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks)):
        result = await coro
        results.append(result)

    return results


class openai_llm:
    def __init__(self, provider="azure", **kwargs):
        """
        Initialize the LLM client.

        Args:
            provider (str): The provider type - "azure" or "vllm"
            **kwargs: Additional arguments for specific providers
                For vllm: base_url, api_key, model_name are required
                For azure: Uses environment variables by default
        """
        load_dotenv()

        self.provider = provider
        self.token_log_file = "./logs/token.json"
        self.endpoint_manager = None  # Will be set up for vLLM

        if provider == "azure":
            # Azure OpenAI settings
            self.endpoint = os.getenv("OPENAI_ENDPOINT")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.api_version = "2024-02-15-preview"
            self.deployment = "gpt-4o-mini-0718"
            self.model = "gpt-4o-mini-0718"

            self.client = AzureOpenAI(
                azure_deployment=self.deployment,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
            self.async_client = AsyncAzureOpenAI(
                azure_deployment=self.deployment,
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        elif provider == "vllm":
            # vLLM with OpenAI-compatible API settings
            self.base_url = kwargs.get("base_url", "http://localhost:8000/v1")
            self.api_key = kwargs.get("api_key", None)  # Make API key optional
            self.model = kwargs.get(
                "model_name", "NousResearch/Meta-Llama-3-8B-Instruct"
            )

            print(
                f"Creating VLLM client with base_url: {self.base_url}, model_name: {self.model}, api_key: {self.api_key}"
            )

            # Configure HTTP client with proper timeouts and connection limits
            timeout = httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=60.0,     # Read timeout
                write=10.0,    # Write timeout
                pool=300.0     # Pool timeout (overall timeout)
            )
            
            limits = httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            )
            
            # Create HTTP client with proper configuration
            http_client = httpx.Client(timeout=timeout, limits=limits)
            async_http_client = httpx.AsyncClient(timeout=timeout, limits=limits)

            # Initialize client without api_key if it's None
            api_key_args = {} if self.api_key is None else {"api_key": self.api_key}
            self.client = OpenAI(
                base_url=self.base_url, 
                http_client=http_client,
                max_retries=2,  # Limit retries at client level
                **api_key_args
            )
            self.async_client = AsyncOpenAI(
                base_url=self.base_url, 
                http_client=async_http_client,
                max_retries=2,  # Limit retries at client level
                **api_key_args
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'azure' or 'vllm'")

    def setup_endpoint_manager(self, endpoints: List[str], circuit_config: Optional[CircuitBreakerConfig] = None):
        """Setup endpoint manager for vLLM endpoints with circuit breaker support."""
        if self.provider == "vllm":
            self.endpoint_manager = EndpointManager(
                endpoints=endpoints,
                circuit_config=circuit_config or CircuitBreakerConfig()
            )
            logger.info(f"Setup endpoint manager for {len(endpoints)} vLLM endpoints")
        else:
            logger.warning("Endpoint manager only supported for vLLM provider")
    
    def get_endpoint_health(self) -> Optional[dict]:
        """Get health statistics for all endpoints."""
        if self.endpoint_manager:
            return self.endpoint_manager.get_endpoint_stats()
        return None
    
    def is_endpoint_healthy(self) -> bool:
        """Check if the current endpoint is healthy."""
        if self.endpoint_manager:
            healthy_endpoints = self.endpoint_manager.get_healthy_endpoints()
            return self.base_url in healthy_endpoints
        return True  # Assume healthy if no endpoint manager

    def cal_cost(self, response, **kwargs):
        if not os.path.exists(self.token_log_file):
            with open(self.token_log_file, "w") as f:
                json.dump({"none": "none"}, f)
        with open(self.token_log_file, "r") as f:
            tokens = json.load(f)
            current_model = kwargs.get("model", self.model)
            if current_model not in tokens:
                tokens[current_model] = [0, 0]
            tokens[current_model][0] += response.usage.prompt_tokens
            tokens[current_model][1] += response.usage.completion_tokens
        with open(self.token_log_file, "w") as f:
            json.dump(tokens, f)

    def cal_batch_cost(self, prompt_tokens, completion_tokens, **kwargs):
        if not os.path.exists(self.token_log_file):
            with open(self.token_log_file, "w") as f:
                json.dump({"none": "none"}, f)
        with open(self.token_log_file, "r") as f:
            tokens = json.load(f)
            current_model = kwargs.get("model", self.model)
            if current_model not in tokens:
                tokens[current_model] = [0, 0]
            tokens[current_model][0] += prompt_tokens
            tokens[current_model][1] += completion_tokens
        with open(self.token_log_file, "w") as f:
            json.dump(tokens, f)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),  # Reduced from 5 to 3 attempts
        before=before_retry_fn,
        retry=retry_if_exception_type((
            TimeoutError, ConnectionError, httpx.TimeoutException, httpx.ConnectError,
            APITimeoutError, APIConnectionError, RateLimitError
        ))
    )
    def response(self, messages, **kwargs):
        model = kwargs.get("model", self.model)
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                n=kwargs.get("n", 1),
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 4000),
                timeout=kwargs.get("timeout", 180),
            )
            
            # Record success if endpoint manager is available
            if self.endpoint_manager and self.provider == "vllm":
                response_time = time.time() - start_time
                self.endpoint_manager.record_request_success(self.base_url, response_time)
            
            # self.cal_cost(response,**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            # Record failure if endpoint manager is available
            if self.endpoint_manager and self.provider == "vllm":
                self.endpoint_manager.record_request_failure(self.base_url)
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),  # Reduced from 5 to 3 attempts
        before=before_retry_fn,
        retry=retry_if_exception_type((
            TimeoutError, ConnectionError, httpx.TimeoutException, httpx.ConnectError,
            APITimeoutError, APIConnectionError, RateLimitError
        ))
    )
    async def response_async(self, messages, **kwargs):
        model = kwargs.get("model", self.model)
        start_time = time.time()

        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                n=kwargs.get("n", 1),
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 4096),
                timeout=kwargs.get("timeout", 180),
            )
            
            # Record success if endpoint manager is available
            if self.endpoint_manager and self.provider == "vllm":
                response_time = time.time() - start_time
                self.endpoint_manager.record_request_success(self.base_url, response_time)
            
            # self.cal_cost(response,**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            # Record failure if endpoint manager is available
            if self.endpoint_manager and self.provider == "vllm":
                self.endpoint_manager.record_request_failure(self.base_url)
            raise

    def generate_output(self, messages, **kwargs):
        try:
            response = self.response(messages, **kwargs)
        except (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.ConnectError,
                APITimeoutError, APIConnectionError, RateLimitError) as e:
            response = "<judge>1</judge>"  # if failed, return not match
            logger.error(f"Retryable error for {kwargs.get('model', self.model)}: {type(e).__name__}: {e}")
        except APIError as e:
            # Non-retryable API errors (like invalid request format, auth issues, etc.)
            response = "<judge>1</judge>"  # if failed, return not match
            logger.error(f"API error for {kwargs.get('model', self.model)}: {e}")
        except Exception as e:
            # Catch-all for truly unexpected errors, but log them distinctly
            response = "<judge>1</judge>"  # if failed, return not match
            logger.error(f"Unexpected error for {kwargs.get('model', self.model)}: {type(e).__name__}: {e}")
        return response

    async def generate_output_async(self, idx, messages, **kwargs):
        try:
            response = await self.response_async(messages, **kwargs)
        except (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.ConnectError,
                APITimeoutError, APIConnectionError, RateLimitError) as e:
            response = "<judge>1</judge>"  # if failed, return not match
            logger.error(f"Retryable error for {kwargs.get('model', self.model)} at index {idx}: {type(e).__name__}: {e}")
        except APIError as e:
            # Non-retryable API errors (like invalid request format, auth issues, etc.)
            response = "<judge>1</judge>"  # if failed, return not match
            logger.error(f"API error for {kwargs.get('model', self.model)} at index {idx}: {e}")
        except Exception as e:
            # Catch-all for truly unexpected errors, but log them distinctly
            response = "<judge>1</judge>"  # if failed, return not match
            logger.error(f"Unexpected error for {kwargs.get('model', self.model)} at index {idx}: {type(e).__name__}: {e}")
        return idx, response

    def generate_outputs(self, messages, **kwargs):
        tasks = [
            self.generate_output_async(i, messages[i], **kwargs)
            for i in range(len(messages))
        ]
        results = asyncio.run(deal_tasks(tasks))
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results

    async def generate_outputs_async(self, messages, **kwargs):
        """Asynchronously generate outputs for a batch of messages.

        Args:
            messages: List of message arrays to process
            **kwargs: Additional arguments for the model

        Returns:
            List of response contents
        """
        tasks = [
            self.generate_output_async(i, messages[i], **kwargs)
            for i in range(len(messages))
        ]
        results = await deal_tasks(tasks)
        results = sorted(results, key=lambda x: x[0])
        results = [x[1] for x in results]
        return results


# Default instance using Azure OpenAI
# judger = openai_llm()

if __name__ == "__main__":

    # Test with Azure OpenAI
    messages = [[{"role": "user", "content": "how are you"}] for _ in range(3)]
    llm = openai_llm()
    results = llm.generate_outputs(messages)
    print("Azure OpenAI results:", results)

    # Test with vLLM
    try:
        llm_vllm = openai_llm(
            provider="vllm",
            base_url="http://localhost:8000/v1",  # Change this to the actual vLLM server address
            model_name="NousResearch/Meta-Llama-3-8B-Instruct",  # Change this to the actual model name
            # No API key needed when server doesn't require authentication
        )
        vllm_messages = [[{"role": "user", "content": "Tell me a short joke"}]]
        vllm_results = llm_vllm.generate_outputs(vllm_messages)
        print("vLLM results:", vllm_results)
    except Exception as e:
        print(f"vLLM test failed (this is expected if vLLM server is not running): {e}")
