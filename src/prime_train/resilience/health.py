"""
Training health monitoring via WandB.

Checks training progress and detects issues.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class TrainingStatus:
    """Status of a training run."""
    healthy: bool
    current_step: int
    last_update: datetime
    reason: Optional[str] = None
    run_id: Optional[str] = None
    run_url: Optional[str] = None


def check_training_status(
    project: Optional[str] = None,
    run_id: Optional[str] = None,
    stall_threshold_minutes: int = 30,
) -> TrainingStatus:
    """
    Check training health via WandB.

    Args:
        project: WandB project name
        run_id: Specific run ID (optional, uses latest if not provided)
        stall_threshold_minutes: Minutes without progress before considering stalled

    Returns:
        TrainingStatus with health information
    """
    try:
        import wandb

        # Initialize API
        api = wandb.Api()

        # Get run
        if run_id:
            run = api.run(f"{project}/{run_id}")
        else:
            # Get most recent run in project
            runs = api.runs(project, order="-created_at", per_page=1)
            if not runs:
                return TrainingStatus(
                    healthy=False,
                    current_step=0,
                    last_update=datetime.now(),
                    reason="No runs found in project",
                )
            run = runs[0]

        # Get latest metrics
        history = run.history(samples=1)
        if history.empty:
            return TrainingStatus(
                healthy=False,
                current_step=0,
                last_update=datetime.now(),
                reason="No training data logged yet",
                run_id=run.id,
                run_url=run.url,
            )

        # Check for stall
        last_step = int(history["_step"].iloc[-1])
        last_update = datetime.fromisoformat(run.updated_at.replace("Z", "+00:00"))

        stall_threshold = timedelta(minutes=stall_threshold_minutes)
        is_stalled = datetime.now(last_update.tzinfo) - last_update > stall_threshold

        if is_stalled:
            return TrainingStatus(
                healthy=False,
                current_step=last_step,
                last_update=last_update,
                reason=f"No progress for {stall_threshold_minutes} minutes",
                run_id=run.id,
                run_url=run.url,
            )

        # Check run state
        if run.state == "crashed":
            return TrainingStatus(
                healthy=False,
                current_step=last_step,
                last_update=last_update,
                reason="Run crashed",
                run_id=run.id,
                run_url=run.url,
            )

        if run.state == "failed":
            return TrainingStatus(
                healthy=False,
                current_step=last_step,
                last_update=last_update,
                reason="Run failed",
                run_id=run.id,
                run_url=run.url,
            )

        return TrainingStatus(
            healthy=True,
            current_step=last_step,
            last_update=last_update,
            run_id=run.id,
            run_url=run.url,
        )

    except ImportError:
        return TrainingStatus(
            healthy=False,
            current_step=0,
            last_update=datetime.now(),
            reason="wandb not installed",
        )
    except Exception as e:
        return TrainingStatus(
            healthy=False,
            current_step=0,
            last_update=datetime.now(),
            reason=f"Error checking status: {e}",
        )
