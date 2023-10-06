"""
This is a boilerplate test file for pipeline 'combine_text'
generated using Kedro 0.18.11.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import pytest
from kedro.pipeline import Pipeline

from mlops_template.pipelines.combine_text.pipeline import (
    combine_title_and_desc,
    create_pipeline,
)


@pytest.fixture
def sample_data():
    # Create some sample data for testing
    data = {
        "title": ["Title 1", "Title 2", "Title 3"],
        "description": ["Description 1", "Description 2", "Description 3"],
    }
    return pd.DataFrame(data)


def test_combine_title_and_desc(sample_data):
    # Call the function with the sample data
    result_df = combine_title_and_desc(sample_data)

    # Check if the result is correct
    expected_data = {
        "text": [
            "Title 1 Description 1",
            "Title 2 Description 2",
            "Title 3 Description 3",
        ]
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_create_pipeline():
    # Call the create_pipeline function
    pipeline = create_pipeline()

    # Check if the pipeline is valid and has the expected structure
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 1

    # Check the properties of the node in the pipeline
    node_params = pipeline.nodes[0]
    assert node_params.func == combine_title_and_desc
    assert node_params.inputs == ["projects_raw"]
    assert node_params.outputs == ["projects_text"]
    assert node_params.name == "combine_title_and_desc_node"
