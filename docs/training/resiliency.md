# Resiliency

Megatron Bridge incorporates resilient training features from the [NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext). This extension provides fault-tolerant capabilities that help minimize downtime due to failures and interruptions during training.

## Fault Tolerance: In Job Restart

The fault tolerance feature can detect hangs during training and automatically restart a workload due to a hang or error. This is particularly useful when training on unreliable hardware, at very large scale, or when transient faults are common.

### Key Features

- **Hang Detection**: Monitors training progress and detects when ranks become unresponsive.
- **Automatic Restart**: Automatically restarts training from the last checkpoint when faults are detected.
- **Section-based Monitoring**: Uses different timeout thresholds for setup, training steps, and checkpointing operations.
- **Timeout Calculation**: Can automatically calculate optimal timeouts based on observed training behavior.
- **Multi-level Restart Logic**: Supports both in-job restarts and new job launches on failure.

### Prerequisites

> **Warning**: This feature is currently only supported on Slurm-based clusters.

Before using fault tolerance features, ensure the following:

1. **Slurm Environment**: The system must be running on a Slurm-based cluster.
2. **Checkpoint Configuration**: A valid directory for saving checkpoints must be properly configured.

### Usage Options

Megatron Bridge provides two ways to enable fault tolerance:

#### Option 1: NeMo Run Plugin

If you're using NeMo Run, the {py:class}`bridge.recipes.run_plugins.FaultTolerancePlugin` provides the simplest integration:

```python
from megatron.bridge.recipes.run_plugins import FaultTolerancePlugin
import nemo_run as run

# Configure your task
task = run.Script(...)

# Add fault tolerance plugin
run_plugins = [
    FaultTolerancePlugin(
        enable_ft_package=True,
        calc_ft_timeouts=True,
        num_in_job_restarts=3,
        num_job_retries_on_failure=2,
        initial_rank_heartbeat_timeout=1800,
        rank_heartbeat_timeout=300,
    )
]

# Run with fault tolerance
run.run(task, plugins=run_plugins, executor=executor)
```

#### Option 2: Direct Configuration

If you’re a user who wants more direct control, you can configure fault tolerance manually:

```python
from megatron.bridge.training.config import FaultToleranceConfig

# Configure fault tolerance in your config
config.ft = FaultToleranceConfig(
    enable_ft_package=True,
    calc_ft_timeouts=True,
    # Optional: simulate faults for testing
    simulate_fault=False,
    simulated_fault_type="random",
)
```

When directly using the configuration, you must launch your training script using the `ft_launcher` tool:

```bash
ft_launcher \
    --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes=${NUM_NODES} --nproc-per-node=${NUM_GPUS_PER_NODE} \
    --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
    --ft-param-rank_out_of_section_timeout=300 \
    your_training_script.py
```

### Configuration Options

The fault tolerance system can be configured through {py:class}`bridge.training.config.FaultToleranceConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_ft_package` | `bool` | `False` | Enable the fault tolerance package |
| `calc_ft_timeouts` | `bool` | `False` | Automatically compute optimal timeouts |
| `simulate_fault` | `bool` | `False` | Enable fault simulation for testing |
| `simulated_fault_type` | `str` | `"random"` | Type of fault to simulate: `"rank_hung"`, `"rank_killed"`, or `"random"` |
| `simulated_fault_rank` | `int` | `None` | Specific rank to simulate fault on (random if not specified) |
| `simulated_fault_base_delay` | `int` | `0` | Base delay before simulating fault |

### Plugin Configuration Options

When using the {py:class}`bridge.recipes.run_plugins.FaultTolerancePlugin`, additional options are available:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_in_job_restarts` | `int` | `3` | Maximum number of restarts within the same job |
| `num_job_retries_on_failure` | `int` | `2` | Maximum number of new job launches on failure |
| `initial_rank_heartbeat_timeout` | `int` | `1800` | Timeout for initial heartbeat (seconds) |
| `rank_heartbeat_timeout` | `int` | `300` | Timeout for subsequent heartbeats (seconds) |

### What to Expect

When fault tolerance is enabled and a hang or fault is detected, you should see log messages similar to:

```
[WARNING] [RankMonitorServer:34] Did not get subsequent heartbeat. Waited 171.92 seconds.
[WARNING] [RankMonitorServer:58] Did not get subsequent heartbeat. Waited 171.92 seconds.
FT: Simulating fault: rank_killed; rank to fail: 2
torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 453152 closing signal SIGTERM
```

The system will then automatically restart training from the most recent checkpoint.

### How It Works

The fault tolerance system integrates with Megatron Bridge's training pipeline through several key points:

1. **Setup Phase**: Initializes fault tolerance monitoring before training begins.
2. **Training Steps**: Wraps each training iteration with timeout monitoring.
3. **Evaluation Steps**: Monitors evaluation iterations separately.
4. **Checkpointing**: Tracks checkpoint saving operations with dedicated timeouts.
5. **State Persistence**: Saves timeout calculations to `ft_state.json` for future runs.

The system uses a section-based approach with different timeout thresholds:
- **Setup Section**: Covers initialization and checkpoint loading.
- **Step Section**: Monitors individual training/evaluation iterations.
- **Checkpointing Section**: Tracks checkpoint saving operations.
- **Out-of-Section**: Handles time between sections.

### Best Practices

1. **Enable Automatic Timeout Calculation**: Set `calc_ft_timeouts=True` to let the system learn optimal timeouts from your workload.
2. **Conservative Restart Limits**: Use reasonable limits for `num_in_job_restarts` and `num_job_retries_on_failure` to avoid infinite restart loops.
3. **Monitor Logs**: Watch for fault tolerance messages to understand when and why restarts occur.
4. **Test with Simulation**: Use the fault simulation features to test your fault tolerance setup before production runs.
5. **Checkpoint Frequency**: Ensure regular checkpointing to minimize lost work during restarts.

### Limitations

- Currently only supported on Slurm-based clusters.
- Not compatible with NSys profiling (the plugin will automatically disable nsys if enabled).
- Checkpoint save directory must be configured and accessible.

## Straggler Detection

The straggler detection feature identifies slow-performing ranks and can optionally terminate training if performance falls below specified thresholds. This helps ensure efficient training by detecting and mitigating the impact of underperforming nodes.

### Key Features

- **Performance Monitoring**: Tracks individual and relative GPU performance scores.
- **Automatic Detection**: Identifies stragglers based on configurable thresholds.
- **Detailed Reporting**: Provides comprehensive performance reports with best/worst performing ranks.
- **Optional Termination**: Can automatically stop training when stragglers are detected.
- **Flexible Configuration**: Supports various reporting intervals and threshold settings.

### Configuration

Enable straggler detection through the {py:class}`bridge.training.config.NVRxStragglerDetectionConfig`:

```python
from megatron.bridge.training.config import NVRxStragglerDetectionConfig

# Configure straggler detection in your config
config.nvrx_straggler = NVRxStragglerDetectionConfig(
    enabled=True,
    report_time_interval=300.0,  # Report every 5 minutes
    calc_relative_gpu_perf=True,
    calc_individual_gpu_perf=True,
    num_gpu_perf_scores_to_print=5,
    gpu_relative_perf_threshold=0.7,
    gpu_individual_perf_threshold=0.7,
    stop_if_detected=False,  # Set to True to stop training on detection
    enable_logging=True,
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable NVRx straggler detection |
| `report_time_interval` | `float` | `300.0` | Interval in seconds between straggler checks |
| `calc_relative_gpu_perf` | `bool` | `True` | Calculate relative GPU performance scores |
| `calc_individual_gpu_perf` | `bool` | `True` | Calculate individual GPU performance scores |
| `num_gpu_perf_scores_to_print` | `int` | `5` | Number of best/worst scores to print (0 disables periodic printing) |
| `gpu_relative_perf_threshold` | `float` | `0.7` | Threshold for relative performance (0.0-1.0) |
| `gpu_individual_perf_threshold` | `float` | `0.7` | Threshold for individual performance (0.0-1.0) |
| `stop_if_detected` | `bool` | `False` | Terminate training if stragglers are detected (saves checkpoint before exiting) |
| `enable_logging` | `bool` | `True` | Log GPU performance scores as structured data |
| `profiling_interval` | `int` | `1` | Profiling interval for the detector |
| `logger_name` | `str` | `"megatron.bridge.NVRxStragglerDetection"` | Logger name for messages |

### Expected Output

When straggler detection is enabled, you'll see performance reports in the training logs similar to:

```
GPU relative performance:
 Worst performing 5/512 ranks:
  Rank=76 Node=h100-001-253-012 Score=0.94
  Rank=13 Node=h100-001-010-003 Score=0.94
  Rank=45 Node=h100-001-172-026 Score=0.94
  Rank=433 Node=h100-004-141-026 Score=0.95
  Rank=308 Node=h100-003-263-012 Score=0.95
 Best performing 5/512 ranks:
  Rank=432 Node=h100-004-141-026 Score=0.99
  Rank=376 Node=h100-004-005-003 Score=0.98
  Rank=487 Node=h100-004-255-026 Score=0.98
  Rank=369 Node=h100-004-004-033 Score=0.98
  Rank=361 Node=h100-004-004-023 Score=0.98

GPU individual performance:
 Worst performing 5/512 ranks:
  Rank=76 Node=h100-001-253-012 Score=0.98
  Rank=162 Node=h100-002-042-026 Score=0.98
  Rank=79 Node=h100-001-253-012 Score=0.98
  Rank=357 Node=h100-004-004-013 Score=0.98
  Rank=85 Node=h100-001-253-026 Score=0.98
 Best performing 5/512 ranks:
  Rank=297 Node=h100-003-095-026 Score=1.00
  Rank=123 Node=h100-001-273-026 Score=1.00
  Rank=21 Node=h100-001-010-013 Score=1.00
  Rank=389 Node=h100-004-074-012 Score=1.00
  Rank=489 Node=h100-004-269-026 Score=1.00

 Straggler report processing time: 0.042 sec.
```

If stragglers are detected and thresholds are exceeded, you'll see warnings like:

```
STRAGGLER DETECTION WARNING: Some GPUs have worse relative performance. Affected ranks: [76, 13, 45]
STRAGGLER DETECTION WARNING: Some GPUs performance dropped. Affected ranks: [162, 79, 357]
```

### Performance Scores

The system calculates two types of performance scores:

1. **Relative Performance**: Compares each rank's performance relative to other ranks in the same training run.
2. **Individual Performance**: Tracks each rank's performance over time to detect degradation.

Scores range from 0.0 to 1.0, where:
- **1.0**: Best possible performance
- **0.7** (default threshold): Below this indicates a potential straggler
- **Lower values**: Indicate worse performance

### How It Works

The straggler detection system:

1. **Initialization**: Sets up the NVRx detector during training setup.
2. **Monitoring**: Wraps the training step function to monitor execution time.
3. **Periodic Reporting**: Generates performance reports at specified intervals.
4. **Straggler Identification**: Compares performance scores against thresholds.
5. **Action**: Optionally saves a checkpoint and terminates training if stragglers are detected.

### Best Practices

1. **Appropriate Intervals**: Set `report_time_interval` based on your training characteristics.
2. **Threshold Tuning**: Adjust thresholds based on your hardware and expected performance variability.
3. **Gradual Rollout**: Start with `stop_if_detected=False` to observe performance patterns before enabling automatic termination.
4. **Monitor Logs**: Regularly check straggler reports to identify persistent hardware issues.
5. **Performance Impact**: The overhead is minimal, but you can adjust `profiling_interval` if needed.

### Integration with Training

The straggler detection integrates directly with the training loop:

- Automatically initializes when {py:class}`bridge.training.resiliency.NVRxStragglerDetectionManager` is configured.
- Monitors training steps without affecting the training logic.
- Provides exit conditions that the training loop respects.
- Safely shuts down when training completes.

## Preemption

Training foundation models can take several hours or even days to complete. In some cases, training jobs must be halted preemptively due to cluster time limits, higher priority jobs, or other reasons.

Megatron Bridge provides functionality to gracefully perform preemptive shutdown of training. This feature listens for user-specified signals and saves a checkpoint before exiting when the signal is received.

### Key Features

- **Signal-based Shutdown**: Listens for signals (default: SIGTERM) during training.
- **Graceful Exit**: Saves checkpoint before terminating to preserve training progress.
- **Distributed Coordination**: Ensures all ranks receive and handle the signal properly.
- **Flexible Configuration**: Supports different signals and timing configurations.

### Usage Options

Megatron Bridge provides two ways to enable preemption handling:

#### Option 1: NeMo Run Plugin (Recommended)

> **Warning**: This plugin is currently only supported on Slurm-based clusters.

If you're using NeMo Run, the {py:class}`bridge.recipes.run_plugins.PreemptionPlugin` provides the simplest integration:

```python
from megatron.bridge.recipes.run_plugins import PreemptionPlugin
import nemo_run as run

# Configure your task
task = run.Script(...)

# Add preemption plugin
run_plugins = [
    PreemptionPlugin(
        preempt_time=60,  # Send signal 60 seconds before time limit
        enable_exit_handler=True,
        enable_exit_handler_for_data_loader=False,
    )
]

# Run with preemption support
run.run(task, plugins=run_plugins, executor=executor)
```

#### Option 2: Direct Configuration

Configure preemption handling directly in your training configuration:

```python
from megatron.bridge.training.config import TrainingConfig
import signal

# Configure preemption in training config
config.train = TrainingConfig(
    exit_signal_handler=True,
    exit_signal=signal.SIGTERM,  # Signal to listen for
    exit_signal_handler_for_dataloader=False,
    # ... other training config options
)
```

### Configuration Options

#### PreemptionPlugin Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preempt_time` | `int` | `60` | Time in seconds before job limit to send preemption signal |
| `enable_exit_handler` | `bool` | `True` | Enable the exit signal handler in training |
| `enable_exit_handler_for_data_loader` | `bool` | `False` | Enable signal handler for dataloader workers |

#### Training Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exit_signal_handler` | `bool` | `False` | Enable signal handler for graceful shutdown |
| `exit_signal` | `int` | `signal.SIGTERM` | Signal to listen for (default: SIGTERM) |
| `exit_signal_handler_for_dataloader` | `bool` | `False` | Enable signal handler for dataloader workers |

### Expected Behavior

When a preemption signal is received, you'll see log messages similar to:

```
Received signal 15, initiating graceful stop
Signal handler installed for 15
exiting program after receiving SIGTERM.
```

The system will:
1. **Detect the signal** at the end of the current training step.
2. **Save a checkpoint** to preserve training progress.
3. **Log the shutdown reason** for debugging purposes.
4. **Exit gracefully** with proper cleanup.

### How It Works

The preemption system operates through several components:

1. **Signal Handler Installation**: Sets up a distributed signal handler using {py:class}`bridge.training.resiliency.DistributedSignalHandler`.
2. **Signal Detection**: Checks for received signals at the end of each training step.
3. **Distributed Coordination**: Uses all-gather to ensure all ranks are aware of the signal.
4. **Checkpoint Saving**: Automatically saves a checkpoint before exiting.
5. **Graceful Shutdown**: Properly cleans up resources and exits.

### Signal Handling Details

The `DistributedSignalHandler` class provides:
- **Cross-rank coordination**: Ensures all ranks handle the signal consistently.
- **Original handler preservation**: Restores original signal handlers on exit.
- **Flexible signal support**: Can handle different signal types (SIGTERM, SIGINT, etc.).

### Integration with Slurm

When using Slurm, the system automatically:
- **Receives SIGTERM** when approaching job time limits.
- **Coordinates across nodes** to ensure consistent shutdown.
- **Saves progress** before the job is forcibly terminated.

### Best Practices

1. **Use Appropriate Timing**: Set `preempt_time` to allow sufficient time for checkpoint saving.
2. **Monitor Logs**: Watch for preemption messages to understand shutdown patterns.
3. **Test Signal Handling**: Verify preemption works correctly in your environment.
4. **Regular Checkpointing**: Ensure regular checkpoint intervals to minimize potential data loss.
5. **Resource Cleanup**: The system handles cleanup automatically, but monitor for any resource leaks.

## Re-run State Machine

The re-run state machine is an experimental feature that helps with attribution of unexpected results such as NaN values, spiky loss, or other computational anomalies. It works by re-running computations to determine whether issues are transient errors, persistent hardware faults, or actually correct results.

> **Disclaimer**: This is an experimental alpha-level feature for result attribution. Nodes flagged by this system should be subjected to standard diagnostic test suites for confirmation.

### Key Features

- **Automatic Re-run Logic**: Detects unexpected results and automatically re-runs computations to verify reproducibility.
- **Error Attribution**: Classifies issues as transient errors, persistent errors, or correct results.
- **Multi-stage Validation**: Uses in-place re-runs and checkpoint-based re-runs on different hardware.
- **Determinism Tracking**: Can report statistics on computational non-determinism.
- **State Management**: Handles RNG state and data iterator state for reproducible re-runs.

### How It Works

The re-run state machine operates through several stages:

1. **Initial Run**: Executes the training step normally, validating results.
2. **First Re-run (In-place)**: If validation fails, re-runs on the same GPU to check reproducibility.
3. **Second Re-run (Different GPU)**: If the issue is reproducible, saves checkpoint and re-runs on different hardware.
4. **Attribution**: Determines if the issue is a transient error, persistent error, or correct result.

### Configuration

Configure the re-run state machine through {py:class}`bridge.training.config.RerunStateMachineConfig`:

```python
from megatron.bridge.training.config import RerunStateMachineConfig

# Configure re-run state machine in your config
config.rerun_state_machine = RerunStateMachineConfig(
    rerun_mode="validate_results",  # or "report_stats" or "disabled"
    check_for_nan_in_loss=True,
    check_for_spiky_loss=False,
    error_injection_rate=0,  # For testing only
    error_injection_type="transient_error",
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rerun_mode` | `str` | `"disabled"` | Operating mode: `"disabled"`, `"validate_results"`, or `"report_stats"` |
| `check_for_nan_in_loss` | `bool` | `True` | Check for NaN values in loss |
| `check_for_spiky_loss` | `bool` | `False` | Check for unexpectedly large loss values |
| `error_injection_rate` | `int` | `0` | Rate for injecting test errors (testing only) |
| `error_injection_type` | `str` | `"transient_error"` | Type of error to inject for testing |

### Operating Modes

#### 1. Disabled Mode (`disabled`)
- **Purpose**: No result validation or re-run logic.
- **Behavior**: Training proceeds normally without any result checking.
- **Use Case**: When re-run overhead is not acceptable or validation is not needed.

#### 2. Report Stats Mode (`report_stats`)  
- **Purpose**: Collect statistics on computational determinism.
- **Behavior**: Re-runs every step once to measure variability.
- **Output**: Reports on computational non-determinism without stopping training.

#### 3. Validate Results Mode (`validate_results`)
- **Purpose**: Full validation with re-runs and hardware fault attribution.
- **Behavior**: Re-runs computations when unexpected results are detected.
- **Exit Conditions**: May exit with specific codes for checkpointing or validation failure.

### Integration with Training

The re-run state machine integrates at the training step level:

```python
# In train_step function
rerun_state_machine = get_rerun_state_machine()
while rerun_state_machine.should_run_forward_backward(data_iterator):
    # Execute forward-backward pass
    loss_dict = forward_backward_func(...)
    
    # Validate results (automatically handled in forward_step)
    # check_for_nan_in_loss and check_for_spiky_loss are passed to loss function

should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
if should_checkpoint:
    save_checkpoint(...)
if should_exit:
    sys.exit(exit_code)
```

### Exit Codes

The re-run state machine uses specific exit codes to control job behavior:

- **Exit Code 16** (`EXIT_CODE_RESUME_TO_DISAMBIGUATE`): Job should be restarted from checkpoint to re-run on different hardware.
- **Exit Code 17** (`EXIT_CODE_FAILED_ON_RESULT_VALIDATION`): Job failed validation and should not continue.

### Expected Behavior

#### Validation Success
When validation passes, training continues normally with no additional overhead.

#### Transient Error Detection
```
Unexpected result tensor(nan) on rank 0 at iteration #150 invocation #1 (message='loss is NaN')
First rerun: unexpected result is not reproducible within the tolerance
Possible transient error!
```

#### Persistent Error Detection  
```
First rerun: unexpected result is reproducible within the tolerance
Need to rerun on a different GPU to verify correctness
Second rerun: unexpected result is not reproducible on a different GPU, therefore was likely incorrect
Possible persistent error!
```

#### Correct Result (False Positive)
```
Second rerun: unexpected result is reproducible on a different GPU, therefore it was likely correct
Correct result (but possible Application error)
```

### Result Attribution Categories

1. **Transient Error**: Result not reproducible on same GPU - likely temporary hardware glitch.
2. **Persistent Error**: Result reproducible on same GPU but different on other GPU - likely hardware fault.
3. **Correct Result**: Result reproducible across different GPUs - likely correct but unexpected.

### Data Iterator Integration

The system uses `RerunDataIterator` to handle data replay:
- **State Saving**: Captures data iterator state for reproducible re-runs.
- **Replay Capability**: Can rewind and replay the same data batches.
- **Checkpoint Support**: Saves/restores iterator state across job restarts.
