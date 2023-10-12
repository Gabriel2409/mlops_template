"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from mlops_template.pipelines import (
    combine_text,
    create_azure_dataasset_from_local_files,
    dummy_classifier,
    finetuned_llm_classifier,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    pipelines["__default__"] = finetuned_llm_classifier.create_pipeline()
    return pipelines
