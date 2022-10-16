import os
import azureml.core
from azureml.core import (
    Workspace,
    Experiment,
    Dataset,
    Datastore,
    ComputeTarget,
    Environment,
    ScriptRunConfig
)
from azureml.data import OutputFileDatasetConfig
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException

"""
Error: Import "azureml.pipeline.steps" could not be resolvedPylancereportMissingImports

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
"""

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)