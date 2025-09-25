"""Resilience utilities for vLLM judge system including circuit breaker and health checks."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
import logging
from collections import deque

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes in half-open before closing
    timeout: float = 60.0  # Seconds to wait before half-open
    window_size: int = 10  # Size of sliding window for tracking


@dataclass
class EndpointHealth:
    """Health status of an endpoint."""
    url: str
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


class CircuitBreaker:
    """Circuit breaker for individual endpoints."""
    
    def __init__(self, endpoint: str, config: CircuitBreakerConfig):
        self.endpoint = endpoint
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_window = deque(maxlen=config.window_size)
        
    def record_success(self):
        """Record a successful request."""
        self.request_window.append(True)
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
            
    def record_failure(self):
        """Record a failed request."""
        self.request_window.append(False)
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._open()
        elif self.state == CircuitState.HALF_OPEN:
            self._open()
            
    def can_request(self) -> bool:
        """Check if requests are allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._half_open()
                return True
            return False
        else:  # HALF_OPEN
            return True
            
    def _open(self):
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.warning(f"Circuit breaker opened for endpoint: {self.endpoint}")
        
    def _close(self):
        """Close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker closed for endpoint: {self.endpoint}")
        
    def _half_open(self):
        """Put circuit in half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit breaker half-open for endpoint: {self.endpoint}")
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try again."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.timeout
        )


class EndpointManager:
    """Manages multiple vLLM endpoints with health checks and circuit breakers."""
    
    def __init__(
        self,
        endpoints: List[str],
        circuit_config: Optional[CircuitBreakerConfig] = None,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 10.0
    ):
        self.endpoints = endpoints
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        
        # Initialize circuit breakers and health status
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            endpoint: CircuitBreaker(endpoint, self.circuit_config)
            for endpoint in endpoints
        }
        self.endpoint_health: Dict[str, EndpointHealth] = {
            endpoint: EndpointHealth(url=endpoint)
            for endpoint in endpoints
        }
        
        # Start health check task
        self._health_check_task = None
        
    async def start_health_checks(self):
        """Start periodic health checks."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
    async def stop_health_checks(self):
        """Stop health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
    async def _health_check_loop(self):
        """Continuously check endpoint health."""
        while True:
            try:
                await self._check_all_endpoints()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
                
    async def _check_all_endpoints(self):
        """Check health of all endpoints."""
        tasks = [
            self._check_endpoint_health(endpoint)
            for endpoint in self.endpoints
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _check_endpoint_health(self, endpoint: str):
        """Check health of a single endpoint."""
        health = self.endpoint_health[endpoint]
        health.last_check = datetime.now()
        
        try:
            # Simple health check - could be replaced with actual API call
            # For now, we'll consider endpoint healthy if circuit breaker allows requests
            circuit_breaker = self.circuit_breakers[endpoint]
            health.is_healthy = circuit_breaker.can_request()
            
            if health.is_healthy and health.consecutive_failures > 0:
                health.consecutive_failures = 0
                logger.info(f"Endpoint {endpoint} is healthy again")
                
        except Exception as e:
            health.is_healthy = False
            health.consecutive_failures += 1
            health.last_failure = datetime.now()
            logger.error(f"Health check failed for {endpoint}: {e}")
            
    def get_healthy_endpoints(self) -> List[str]:
        """Get list of currently healthy endpoints."""
        return [
            endpoint for endpoint, health in self.endpoint_health.items()
            if health.is_healthy and self.circuit_breakers[endpoint].can_request()
        ]
        
    def record_request_success(self, endpoint: str, response_time: float):
        """Record successful request."""
        if endpoint in self.circuit_breakers:
            self.circuit_breakers[endpoint].record_success()
            health = self.endpoint_health[endpoint]
            health.total_requests += 1
            health.response_times.append(response_time)
            
    def record_request_failure(self, endpoint: str):
        """Record failed request."""
        if endpoint in self.circuit_breakers:
            self.circuit_breakers[endpoint].record_failure()
            health = self.endpoint_health[endpoint]
            health.total_requests += 1
            health.failed_requests += 1
            health.consecutive_failures += 1
            health.last_failure = datetime.now()
            
    def get_endpoint_stats(self) -> Dict[str, dict]:
        """Get statistics for all endpoints."""
        stats = {}
        for endpoint, health in self.endpoint_health.items():
            circuit = self.circuit_breakers[endpoint]
            stats[endpoint] = {
                "healthy": health.is_healthy,
                "circuit_state": circuit.state.value,
                "success_rate": health.success_rate,
                "avg_response_time": health.avg_response_time,
                "total_requests": health.total_requests,
                "failed_requests": health.failed_requests,
                "consecutive_failures": health.consecutive_failures,
            }
        return stats


async def exponential_backoff_retry(
    func,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
    """
    import random
    
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            
            if attempt == max_attempts - 1:
                raise
                
            # Calculate delay with exponential backoff
            delay = min(initial_delay * (exponential_base ** attempt), max_delay)
            
            # Add jitter if requested
            if jitter:
                delay *= (0.5 + random.random())
                
            logger.warning(
                f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                f"Retrying in {delay:.2f} seconds..."
            )
            
            await asyncio.sleep(delay)
            
    raise last_exception