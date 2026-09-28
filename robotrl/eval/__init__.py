"""评估层：把"效果好不好"变成可比较的数字。

强化学习的实验结论很容易不可靠——换个随机种子、换个评估回合数，均值能差
出好几个身位。本层用三件事把不确定性压下去：

- 训练与评估分离，评估用固定种子、固定回合数、确定性动作；
- 结果带标准差，不报单次最好成绩；
- 基线落盘成 JSON，下次不重跑也能对比。

外部只需调用 evaluate_policy() 拿指标，或用 save_report()/load_report()
读写基线。
"""

from robotrl.eval.harness import (
    FINAL_SEEDS,
    TRAIN_SEED,
    EpisodeResult,
    EvalProtocol,
    EvalResult,
    SegmentStats,
    evaluate_policy,
    evaluate_with_protocol,
    final_protocol,
    train_protocol,
)
from robotrl.eval.report import (
    Report,
    compare_reports,
    flatten_metrics,
    load_report,
    save_report,
)

__all__ = [
    "FINAL_SEEDS",
    "TRAIN_SEED",
    "EpisodeResult",
    "EvalProtocol",
    "EvalResult",
    "Report",
    "SegmentStats",
    "compare_reports",
    "evaluate_policy",
    "evaluate_with_protocol",
    "final_protocol",
    "flatten_metrics",
    "load_report",
    "save_report",
    "train_protocol",
]
