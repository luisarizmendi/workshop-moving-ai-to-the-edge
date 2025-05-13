from kfp import dsl, compiler, kubernetes
import os
from typing import NamedTuple

# Component 1: Download Dataset
@dsl.component(
    base_image="quay.io/luisarizmendi/pytorch-custom:latest",
    packages_to_install=["roboflow", "pyyaml"]
)
def download_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    dataset_path: dsl.OutputPath(str)
) -> None:
    from roboflow import Roboflow
    import yaml
    import os

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version = project.version(version)
    dataset = version.download("yolov11")

    # Update data.yaml paths
    dataset_yaml_path = f"{dataset.location}/data.yaml"
    with open(dataset_yaml_path, "r") as file:
        data_config = yaml.safe_load(file)

    data_config["train"] = f"{dataset.location}/train/images"
    data_config["val"] = f"{dataset.location}/valid/images"
    data_config["test"] = f"{dataset.location}/test/images"


    print(dataset)



    with open(dataset_path, "w") as f:
        f.write(dataset.location)


# Component 2: Train Model
@dsl.component(
    base_image="quay.io/luisarizmendi/pytorch-custom:latest",
    packages_to_install=["ultralytics", "torch", "pandas"]
)
def train_model(
    dataset_path: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    name: str = "yolo",
    yolo_model: str = "yolo11m.pt",
    optimizer: str = "SGD",
    learning_rate: float = 0.005,
) -> NamedTuple('Outputs', [
    ('train_dir', str),
    ('test_dir', str),
    ('metrics', dict),
    ('inference_outputdims', str)
]):
    import torch
    from ultralytics import YOLO
    import pandas as pd
    import os
    import onnx
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    CONFIG = {
        'name': name,
        'model': yolo_model,
        'data': f"{dataset_path}/data.yaml",
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'optimizer': optimizer,
        'lr0': 0.001,
        'lrf': learning_rate,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_bias_lr': 0.01,
        'warmup_momentum': 0.8,
        'amp': False,
    }

    # Configure PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Initialize and train model
    model = YOLO(CONFIG['model'])
    results_train = model.train(
        name=CONFIG['name'],
        data=CONFIG['data'],
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch'],
        imgsz=CONFIG['imgsz'],
        device=CONFIG['device'],
        
        # Optimizer parameters
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'],
        warmup_epochs=CONFIG['warmup_epochs'],
        warmup_bias_lr=CONFIG['warmup_bias_lr'],
        warmup_momentum=CONFIG['warmup_momentum'],
        amp=CONFIG['amp'],
    )
    
    # Evaluate model
    results_test = model.val(
        data=CONFIG['data'],
        split='test',
        device=CONFIG['device'],
        imgsz=CONFIG['imgsz']
    )

    # Export to ONNX format
    export_path = model.export(format='onnx', imgsz=640, dynamic=True)
    onnx_model = onnx.load(export_path)
    output_tensor = onnx_model.graph.output[0]
    inference_outputdims = [
        d.dim_value if (d.dim_value > 0) else -1
        for d in output_tensor.type.tensor_type.shape.dim
    ]
    print("Exported model output shape:", inference_outputdims)

    # Compute metrics from CSV
    results_csv_path = os.path.join(results_train.save_dir, "results.csv")
    results_df = pd.read_csv(results_csv_path)

    # Extract metrics
    metrics = {
        "precision": results_df["metrics/precision(B)"].iloc[-1],
        "recall": results_df["metrics/recall(B)"].iloc[-1],
        "mAP50": results_df["metrics/mAP50(B)"].iloc[-1],
        "mAP50-95": results_df["metrics/mAP50-95(B)"].iloc[-1]
    }

    return NamedTuple('Outputs', [
        ('train_dir', str),
        ('test_dir', str),
        ('metrics', dict),
        ('inference_outputdims', str)
    ])(
        train_dir=str(results_train.save_dir),
        test_dir=str(results_test.save_dir),
        metrics=metrics,
        inference_outputdims=str(inference_outputdims)
    )

    
# Component 3: Upload to Object Storage
@dsl.component(
    base_image="quay.io/luisarizmendi/pytorch-custom:latest",
)
def upload_to_storage(
    train_dir: str,
    test_dir: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    outputdims: str,
    model_path: dsl.OutputPath(str)
) -> NamedTuple('Outputs', [
    ('model_artifact_s3_path', str),
    ('files_model', str),
    ('tag', str)
]):
    import boto3
    from botocore.exceptions import NoCredentialsError, PartialCredentialsError
    import os
    from datetime import datetime

    tag=datetime.now().strftime("%m-%d-%H_%M")

    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{endpoint}",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False 
    )

    # Get paths for files
    weights_path = os.path.join(train_dir, "weights")

    files_train = [os.path.join(train_dir, f) for f in os.listdir(train_dir)
                   if os.path.isfile(os.path.join(train_dir, f))]
    files_models = [os.path.join(weights_path, f) for f in os.listdir(weights_path)
                    if os.path.isfile(os.path.join(weights_path, f))]

    files_model = os.path.join(train_dir, "weights") + "/best"
    
    files_test = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                  if os.path.isfile(os.path.join(test_dir, f))]

    directory_name = os.path.basename(train_dir) + "-" + tag

    # Upload files
    for file_path in files_train:
        try:
            s3_client.upload_file(file_path, bucket, f"{directory_name}/metrics/train-val/{os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")

    for file_path in files_test:
        try:
            s3_client.upload_file(file_path, bucket, f"{directory_name}/metrics/test/{os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")

    with open(model_path, "w") as f:
        f.write(directory_name)

    try:
        s3_client.upload_file(f"{files_model}.pt", bucket, f"{directory_name}/{os.path.basename(files_model)}.pt")
    except Exception as e:
        print(f"Error uploading {files_model}.pt: {e}")

    try:
        s3_client.upload_file(f"{files_model}.onnx", bucket, f"{directory_name}/serving/hardhat/1/model.onnx")
    except Exception as e:
        print(f"Error uploading {files_model}.onnx: {e}")

    try:
        # Create the config.pbtxt file
        config_pbtxt = f"""\
name: "hardhat"
platform: "onnxruntime_onnx"
max_batch_size: 0  
input [
{{
    name: "images"
    data_type: TYPE_FP32
    dims: [-1, 3, 640, 640]  
}}
]
output [
{{
    name: "output0"
    data_type: TYPE_FP32
    dims: {outputdims}
}}
]
backend: "onnxruntime"
"""

        with open("config.pbtxt", "w") as f:
            f.write(config_pbtxt)
            
        s3_client.upload_file("config.pbtxt", bucket, f"{directory_name}/serving/hardhat/config.pbtxt")
    except Exception as e:
        print(f"Error uploading config.pbtxt: {e}")

    model_artifact_s3_path = directory_name

    return NamedTuple('Outputs', [
        ('model_artifact_s3_path', str),
        ('files_model', str),
        ('tag', str)
    ])(
        model_artifact_s3_path,
        os.path.basename(files_model),
        tag
    )


# Component 5: Push to Model Registry
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['model-registry']
)
def push_to_model_registry(
    user_name: str,
    model_name: str,
    model_format_name: str,
    metrics: dict,
    model_registry_name: str,
    output_dims: str,
    container_registry: str,
    modelcar_image_name: str,
    modelcar_image_tag: str,
    
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: int,
    train_epochs: int,
    train_batch_size: int,
    train_img_size: int
):
    from model_registry import ModelRegistry
    from model_registry import utils
    import os
    import json
    import re
 
    container_registry_clean = re.sub(r"^https?://([^/]+).*", r"\1", container_registry)
    
    model_object_prefix = model_name if model_name else "model"
    
    # To avoid making the user introduce the cluster domain I get it from the Quay endpoint (that should be running in the same cluster). That's why in the vars I use the external endpoint for Quay
    cluster_domain= ""
    pattern = re.compile(r"apps\.([^/]+)")
    match = re.search(pattern, container_registry)
    cluster_domain = match.group(1) if match else None

    server_address = f"https://{model_registry_name}-rest.apps.{cluster_domain}"
    
    print(f"Publishing model into {server_address}")
    
    #namespace_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    #with open(namespace_file_path, 'r') as namespace_file:
    #    namespace = namespace_file.read().strip()
 
    os.environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"
   
    def _register_model():
        registry = ModelRegistry(server_address=server_address, port=443, author=user_name, is_secure=False)
        registered_model_name = model_object_prefix
        metadata = {
            "Dataset": f"https://universe.roboflow.com/{roboflow_workspace}/{roboflow_project}/dataset/{str(roboflow_version)}",
            "Epochs": str(train_epochs),
            "Batch Size": str(train_batch_size),
            "Image Size": str(train_img_size),
            "mAP50": str(metrics["mAP50"]),
            "mAP50-95": str(metrics["mAP50-95"]),
            "precision": str(metrics["precision"]),
            "recall": str(metrics["recall"]),
            "output dims": str(output_dims)
        }
      
        rm = registry.register_model(
            registered_model_name,
            f"oci://{container_registry_clean}/{user_name}/modelcar-{modelcar_image_name}:{modelcar_image_tag}",
            version=modelcar_image_tag,
            description=f"{registered_model_name} is a dense neural network that detects Hardhats in images.",
            model_format_name=model_format_name,
            model_format_version="1",
            metadata=metadata
        )
        print("Model registered successfully")
    
    _register_model()



# Component 4: Trigger Tekton PipelineRun
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['kubernetes']
)



def create_modelcar(
        pipeline_name: str,
        
        user_name: str,
        
        object_storage_endpoint: str,
        object_storage_bucket: str,
        object_storage_path: str,
        object_storage_access_key: str,
        object_storage_secret_key: str,
        
        modelcar_image_name: str,
        modelcar_image_tag: str,
        
        container_registry_credentials: str,
        container_registry: str,
        
) -> str:
    
    from kubernetes import client, config
    import time
    import random
    import string
    import re

    pipeline_run_name=f"modelcar-run-{modelcar_image_tag}"
    modelcar_image_name=f"modelcar-{modelcar_image_name}"

    # Underscores  are not allowed in k8s names
    pipeline_run_name = pipeline_run_name.replace("_", "-")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=3))
    pipeline_run_name += f"-{random_str}"
 
    container_registry_clean = re.sub(r"^https?://([^/]+).*", r"\1", container_registry)
    print(f"Using this Container Registry: {container_registry_clean}")
 
    config.load_incluster_config()
    custom_api = client.CustomObjectsApi()

    pipeline_run_manifest = {
        "apiVersion": "tekton.dev/v1",
        "kind": "PipelineRun",
        "metadata": {
            "name": pipeline_run_name,
            "namespace": f"{user_name}-tools"
        },
        "spec": {
            "params": [
                {
                    "name": "object-api-url",
                    "value": object_storage_endpoint
                },
                {
                    "name": "username",
                    "value": user_name
                },
                {
                    "name": "object_access_key",
                    "value": object_storage_access_key
                },
                {
                    "name": "object_secret_key",
                    "value": object_storage_secret_key
                },
                {
                    "name": "object-bucket",
                    "value": object_storage_bucket
                },
                {
                    "name": "object-directory-path",
                    "value": f"{object_storage_path}/serving"
                },
                {
                    "name": "container-registry-image-name",
                    "value": modelcar_image_name
                },
                {
                    "name": "container-registry",
                    "value": f"{container_registry_clean}/{user_name}"
                },
                {
                    "name": "container-registry-image-tag",
                    "value": modelcar_image_tag
                }
            ],
            "pipelineRef": {
                "name": pipeline_name
            },
            "taskRunTemplate": {
                "serviceAccountName": "pipeline"
            },
            "timeouts": {
                "pipeline": "1h0m0s"
            },
            "workspaces": [
                {
                    "name": "shared-workspace",
                    "persistentVolumeClaim": {
                        "claimName": "ai-modelcar-pvc"
                    }
                },
                {
                    "name": "podman-credentials",
                    "secret": {
                        "secretName": container_registry_credentials
                    }
                }
            ]
        }
    }

    namespace_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read().strip()


    custom_api.create_namespaced_custom_object(
        group="tekton.dev",
        version="v1",
        namespace=f"{user_name}-tools",
        plural="pipelineruns",
        body=pipeline_run_manifest
    )
    print(f"Tekton PipelineRun '{pipeline_run_name}' triggered for pipeline '{pipeline_name}'.")

    time.sleep(5)

    # Poll
    timeout_seconds = 1800  
    poll_interval = 10
    elapsed = 0

    while elapsed < timeout_seconds:
        time.sleep(poll_interval)
        elapsed += poll_interval

        run = custom_api.get_namespaced_custom_object(
            group="tekton.dev",
            version="v1",
            namespace=f"{user_name}-tools",
            plural="pipelineruns",
            name=pipeline_run_name
        )

        conditions = run.get("status", {}).get("conditions", [])
        if not conditions:
            continue

        condition = conditions[0]
        status = condition.get("status")
        reason = condition.get("reason")
        message = condition.get("message", "")

        if status == "True" and reason == "Succeeded":
            print(f"PipelineRun {pipeline_run_name} succeeded.")
            break
        elif status == "False":
            raise RuntimeError(f"PipelineRun {pipeline_run_name} failed: {reason} - {message}")

    else:
        raise TimeoutError(f"PipelineRun {pipeline_run_name} did not complete within timeout.")

    return pipeline_run_name


    
    
# Define the pipeline
@dsl.pipeline(
    name='YOLO Training Pipeline',
    description='Pipeline to download data, train YOLO model, and upload results to OpenShift Data Foundation'
)
def yolo_training_pipeline(

    roboflow_api_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: int,

    workshop_username: str,
    container_registry: str,
            
    object_storage_bucket: str,
    object_access_key: str,
    object_secret_key: str,
    object_storage_endpoint: str = "s3.openshift-storage.svc:443",
      
    train_name: str = "hardhat",
    train_yolo_model: str = "yolo11m.pt",
    train_optimizer: str = "SGD",
    train_learning_rate: float = 0.005,
    train_epochs: int = 50,
    train_batch_size: int = 16,
    train_img_size: int = 640,
      
    container_registry_secret_name: str = "container-registry-credentials",
    
    model_registry_name: str = "object-detection-model-registry"
):
    
    # Create PV
    pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-kubeflow-pvc",
        access_modes=['ReadWriteOnce'],
        size="5Gi",
        storage_class_name="ocs-storagecluster-ceph-rbd",
    )
    pvc_shm = kubernetes.CreatePVC(
        pvc_name_suffix="shm",
        access_modes=['ReadWriteOnce'],
        size="1Gi",
        storage_class_name="ocs-storagecluster-ceph-rbd",
    )    



    # Download dataset
    download_task = download_dataset(
        api_key=roboflow_api_key,
        workspace=roboflow_workspace,
        project=roboflow_project,
        version=roboflow_version
    )
    download_task.set_caching_options(enable_caching=False)
    download_task.set_accelerator_limit(1)
    download_task.set_accelerator_type("nvidia.com/gpu")
    download_task.add_node_selector_constraint("nvidia.com/gpu")

    kubernetes.mount_pvc(
        download_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/opt/app-root/src',
    )
    kubernetes.add_toleration(
        download_task,
        key="nvidia.com/gpu",
        operator="Equal",       
        value="True",           
        effect="NoSchedule"
    )



    # Train model
    train_task = train_model(
        dataset_path=download_task.output,
        epochs=train_epochs,
        batch_size=train_batch_size,
        img_size=train_img_size,
        name=train_name,
        optimizer=train_optimizer,
        learning_rate=train_learning_rate,
        yolo_model=train_yolo_model
    ).after(download_task)
    train_task.set_accelerator_limit(1)
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.add_node_selector_constraint("nvidia.com/gpu")
    train_task.set_memory_request('2Gi')
    train_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        train_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/opt/app-root/src',
    )
    kubernetes.mount_pvc(
        train_task,
        pvc_name=pvc_shm.outputs['name'],
        mount_path='/dev/shm',
    )
    kubernetes.add_toleration(
        train_task,
        key="nvidia.com/gpu",
        operator="Equal",       
        value="True",           
        effect="NoSchedule"
    )
    
        
    
    # Upload results
    upload_task = upload_to_storage(
        train_dir=train_task.outputs['train_dir'],
        test_dir=train_task.outputs['test_dir'],
        endpoint=object_storage_endpoint,
        access_key=object_access_key,
        secret_key=object_secret_key,
        bucket=object_storage_bucket,
        outputdims=train_task.outputs['inference_outputdims']
    ).after(train_task)
    upload_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        upload_task,
        pvc_name=pvc.outputs['name'],
        mount_path='/opt/app-root/src',
    )
    kubernetes.add_toleration(
        upload_task,
        key="nvidia.com/gpu",
        operator="Equal",       
        value="True",           
        effect="NoSchedule"
    )
    delete_pvc = kubernetes.DeletePVC(
        pvc_name=pvc.outputs['name']
    ).after(upload_task)
    
    delete_pvc_shm = kubernetes.DeletePVC(
        pvc_name=pvc_shm.outputs['name']
    ).after(train_task)

    

    # Create ModelCar

    modelcar_task = create_modelcar(
        pipeline_name="ai-modelcar" ,
      
        user_name=workshop_username,
        
        object_storage_endpoint=object_storage_endpoint,
        object_storage_bucket=object_storage_bucket,
        object_storage_access_key=object_access_key,
        object_storage_secret_key=object_secret_key,
        object_storage_path=upload_task.outputs['model_artifact_s3_path'],
        
        container_registry_credentials=container_registry_secret_name,
        container_registry=container_registry,
        modelcar_image_name=train_name,
        modelcar_image_tag=upload_task.outputs['tag'],
        
    ).after(upload_task)
    modelcar_task.set_caching_options(enable_caching=False)



    # Push to model registry
    push_to_model_registry(
        user_name=workshop_username,
        model_name=train_name,
        model_format_name="ONNX" ,
        metrics=train_task.outputs['metrics'],
        model_registry_name=model_registry_name,
        output_dims=train_task.outputs['inference_outputdims'],
        container_registry=container_registry,
        modelcar_image_name=train_name,
        modelcar_image_tag=upload_task.outputs['tag'],
        
        roboflow_workspace=roboflow_workspace,
        roboflow_project=roboflow_project,
        roboflow_version=roboflow_version,
        train_epochs=train_epochs,
        train_batch_size=train_batch_size,
        train_img_size=train_img_size
    ).after(modelcar_task)
    


if __name__ == "__main__":
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=yolo_training_pipeline,
        package_path='yolo_training_pipeline.yaml'
    )
    print("Pipeline compiled successfully to yolo_training_pipeline.yaml")


