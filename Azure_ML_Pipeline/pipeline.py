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

# Create a workspace object from the existing Azure Machine Learning workspace.
workspace = Workspace.from_config()

# Create a new experiment to hold the pipeline results.
exp = Experiment(workspace=workspace, name="keras-mnist-fashion") # experiment will test on the mnist datasets.

use_gpu = False

# choose a name for your cluster
cluster_name = "gpu-cluster" if use_gpu else "cpu-cluster"

found = False
# Check if this compute target already exists in the workspace.
cts = workspace.compute_targets
if cluster_name in cts and cts[cluster_name].type == "AmlCompute":
    found = True
    print("Found existing compute target.")
    compute_target = cts[cluster_name]
if not found:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size= "STANDARD_NC6" if use_gpu else "STANDARD_D2_V2",
        # vm_priority = 'lowpriority', # optional
        max_nodes=4,
    )

    # Create the cluster.
    compute_target = ComputeTarget.create(workspace, cluster_name, compute_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min_node_count is provided, it will use the scale settings for the cluster.
    compute_target.wait_for_completion(
        show_output=True, min_node_count=None, timeout_in_minutes=10
    )
# For a more detailed view of current AmlCompute status, use get_status().print(compute_target.get_status().serialize())

"""

MNIST dataset contains 60,000 training images and 10,000 test images. Each image is a 28x28 pixel grayscale image, associated with a label from 10 classes. The 10 classes represent articles of clothing. Each training and test example is assigned to one of the following labels:

"""

data_urls = ["https://data4mldemo6150520719.blob.core.windows.net/demo/mnist-fashion"]
fashion_ds = Dataset.File.from_files(data_urls)

# list the files referenced by fashion_ds
print(fashion_ds.to_path())

"""

Create the data-prep pipeline step

convert the compressed data files to csv files.

"""

datastore = workspace.get_default_datastore()
prepared_fashion_ds = OutputFileDatasetConfig(
    destination=(datastore, "outputdataset/mnist-fashion") # output dataset will be stored in the outputdataset folder.
).register_on_complete(name="prepared_fashion_ds")