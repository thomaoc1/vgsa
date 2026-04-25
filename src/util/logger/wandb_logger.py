from typing import override

import wandb

from src.util.logger.common.protocol import LoggerProtocol


class WandbLogger(LoggerProtocol):
    def __init__(
        self,
        experiment_group: str,
        config: dict,
        summary_metrics: dict[str, list[str]] | None = None,
    ) -> None:
        self.run = wandb.init(
            entity="thomaoc-",
            project="vgsa",
            group=experiment_group,
            config=config,
        )

        if summary_metrics:
            for metric, summaries in summary_metrics.items():
                for summary in summaries:
                    wandb.define_metric(metric, summary=summary)

    @override
    def log(self, step_result: dict):
        self.run.log(step_result)

    @override
    def finish(self):
        self.run.finish()
