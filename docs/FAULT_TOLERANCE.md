# Ray Actor Fault Tolerance

EasyR1 includes built-in fault tolerance for Ray actors to ensure robust training in production environments where nodes may fail due to hardware issues, network problems, or other external conditions.

## Features

- **Automatic Actor Restart**: Actors automatically restart when they crash or die
- **Task Retry**: Individual tasks are retried when actors fail during execution
- **Health Monitoring**: Periodic health checks for all actors
- **Metrics and Logging**: Comprehensive monitoring of actor failures and restarts
- **Configurable Parameters**: Fine-tune fault tolerance behavior via configuration

## Configuration

Fault tolerance is configured through the `FaultToleranceConfig` class in your training configuration:

```python
from verl.trainer.config import PPOConfig, FaultToleranceConfig

# Create configuration with fault tolerance
config = PPOConfig()
config.fault_tolerance = FaultToleranceConfig(
    enable_fault_tolerance=True,      # Enable/disable fault tolerance
    max_restarts=3,                   # Max actor restarts (0 = no restarts)
    max_task_retries=2,               # Max task retries on actor death
    enable_health_monitoring=True,    # Enable periodic health checks
    health_check_interval=30.0        # Health check interval in training steps
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_fault_tolerance` | `bool` | `True` | Enable/disable fault tolerance |
| `max_restarts` | `int` | `3` | Maximum number of times to restart an actor when it crashes |
| `max_task_retries` | `int` | `2` | Maximum number of times to retry a task on actor death |
| `enable_health_monitoring` | `bool` | `True` | Enable periodic actor health monitoring |
| `health_check_interval` | `float` | `30.0` | Interval (in training steps) for health checks |

### YAML Configuration

You can also configure fault tolerance via YAML:

```yaml
fault_tolerance:
  enable_fault_tolerance: true
  max_restarts: 3
  max_task_retries: 2
  enable_health_monitoring: true
  health_check_interval: 30.0
```

## How It Works

### Actor Restart Mechanism

When an actor crashes or dies:

1. **Detection**: Ray detects the actor failure and raises a `RayActorError`
2. **Automatic Restart**: If `max_restarts > 0`, Ray automatically creates a new actor instance
3. **State Reset**: The new actor starts with fresh state (no state persistence by default)
4. **Task Retry**: If `max_task_retries > 0`, the failed task is retried on the new actor

### Task Retry Mechanism

When a task fails due to actor death:

1. **Detection**: Ray detects the task failure due to actor death
2. **Actor Restart**: The actor is restarted (if restarts are available)
3. **Task Retry**: The task is retried on the restarted actor
4. **Failure Handling**: If all retries are exhausted, the task fails permanently

### Health Monitoring

The system continuously monitors actor health:

1. **Periodic Checks**: Health checks run every `health_check_interval` steps
2. **Status Reporting**: Health status is logged as metrics
3. **Early Detection**: Helps identify actors that may be struggling before they fail

## Monitoring and Metrics

The fault tolerance system provides comprehensive metrics:

### Actor Failure Metrics

- `actor_failures/actor_rollout`: Number of actor rollout failures
- `actor_failures/critic`: Number of critic failures
- `actor_failures/ref_policy`: Number of reference policy failures
- `actor_failures/total`: Total number of actor failures
- `actor_failures/{actor_type}_cumulative`: Cumulative failures per actor type

### Actor Restart Metrics

- `actor_restarts/actor_rollout`: Number of actor rollout restarts
- `actor_restarts/critic`: Number of critic restarts
- `actor_restarts/ref_policy`: Number of reference policy restarts
- `actor_restarts/total`: Total number of actor restarts
- `actor_restarts/{actor_type}_cumulative`: Cumulative restarts per actor type

### Health Metrics

- `actor_health/actor_rollout`: Health status (1=healthy, 0=unhealthy)
- `actor_health/critic`: Health status (1=healthy, 0=unhealthy)
- `actor_health/ref_policy`: Health status (1=healthy, 0=unhealthy)

## Best Practices

### Production Deployment

1. **Enable Fault Tolerance**: Always enable fault tolerance in production
2. **Appropriate Restart Limits**: Set `max_restarts` to 3-5 for production workloads
3. **Task Retry Configuration**: Use `max_task_retries=2` for most scenarios
4. **Health Monitoring**: Enable health monitoring for proactive failure detection
5. **Monitoring Setup**: Monitor failure and restart metrics to identify systemic issues

### Testing

1. **Chaos Engineering**: Regularly test fault tolerance by killing actors during training
2. **Failure Simulation**: Use `ray.kill()` to simulate actor failures
3. **Monitoring Verification**: Verify that failure and restart metrics are properly logged

### Troubleshooting

1. **Check Logs**: Monitor logs for actor failure and restart messages
2. **Review Metrics**: Check failure and restart metrics for patterns
3. **Health Status**: Monitor health check results for early warning signs
4. **Resource Limits**: Ensure adequate resources for actor restarts

## Example Usage

### Basic Configuration

```python
from verl.trainer.config import PPOConfig

# Use default fault tolerance settings
config = PPOConfig()
# fault_tolerance is enabled by default with sensible defaults

# Train with fault tolerance
trainer = RayPPOTrainer(config=config, ...)
trainer.fit()
```

### Custom Configuration

```python
from verl.trainer.config import PPOConfig, FaultToleranceConfig

# Custom fault tolerance configuration
config = PPOConfig()
config.fault_tolerance = FaultToleranceConfig(
    enable_fault_tolerance=True,
    max_restarts=5,           # More aggressive restart policy
    max_task_retries=3,       # More task retries
    enable_health_monitoring=True,
    health_check_interval=10.0  # More frequent health checks
)

trainer = RayPPOTrainer(config=config, ...)
trainer.fit()
```

### Disabling Fault Tolerance

```python
from verl.trainer.config import PPOConfig, FaultToleranceConfig

# Disable fault tolerance (not recommended for production)
config = PPOConfig()
config.fault_tolerance = FaultToleranceConfig(
    enable_fault_tolerance=False
)

trainer = RayPPOTrainer(config=config, ...)
trainer.fit()
```

## Limitations

1. **State Persistence**: Actor state is not automatically persisted across restarts
2. **Failure Limits**: Actors will permanently fail after `max_restarts` failures
3. **Resource Overhead**: Fault tolerance adds slight overhead to training
4. **Network Partitions**: May not handle network partitions gracefully

## Testing

Run the fault tolerance tests to verify functionality:

```bash
# Unit tests
python -m pytest tests/test_fault_tolerance.py -v

# Integration tests (requires Ray)
python -m pytest tests/test_fault_tolerance_integration.py -v
```

## Migration Guide

If you're upgrading from a version without fault tolerance:

1. **Default Behavior**: Fault tolerance is enabled by default with sensible settings
2. **No Code Changes**: Existing code will automatically use fault tolerance
3. **Configuration**: Optionally customize fault tolerance parameters
4. **Monitoring**: Add monitoring for new fault tolerance metrics

## Advanced Configuration

### Environment Variables

You can also configure fault tolerance via environment variables:

```bash
export VERL_FAULT_TOLERANCE_ENABLED=true
export VERL_MAX_RESTARTS=5
export VERL_MAX_TASK_RETRIES=3
export VERL_HEALTH_MONITORING_ENABLED=true
export VERL_HEALTH_CHECK_INTERVAL=30.0
```

### Programmatic Configuration

```python
import os
from verl.trainer.config import PPOConfig, FaultToleranceConfig

# Configure via environment variables
config = PPOConfig()
config.fault_tolerance = FaultToleranceConfig(
    enable_fault_tolerance=os.getenv('VERL_FAULT_TOLERANCE_ENABLED', 'true').lower() == 'true',
    max_restarts=int(os.getenv('VERL_MAX_RESTARTS', '3')),
    max_task_retries=int(os.getenv('VERL_MAX_TASK_RETRIES', '2')),
    enable_health_monitoring=os.getenv('VERL_HEALTH_MONITORING_ENABLED', 'true').lower() == 'true',
    health_check_interval=float(os.getenv('VERL_HEALTH_CHECK_INTERVAL', '30.0'))
)
```