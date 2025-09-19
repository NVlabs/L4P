# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import paths  # noqa: F401
import time
import torch
from lightning.fabric import Fabric
from l4p.utils.vis import generate_video_visualizations, generate_4D_visualization
from l4p.models.utils import prepare_model
from l4p.utils.viser import visualize_point_cloud_viser
from l4p.data.davis import DavisDataset
from l4p.data.video_dataset import VideoDataset
from l4p.data.dycheck_dataset import DycheckDataset


def main():
    # set the precision, accelerator, and limit_gpu_mem_usage
    precision = "16-mixed"
    accelerator = "gpu"
    limit_gpu_mem_usage = False
    # limit_gpu_mem_usage=True  # set to True if Out of Memory
    vis_list = []
    vis_count = 0
    start_port = 8001

    # load the model
    model_name = "l4p_depth_flow_2d3dtrack_camray_dynseg_v1"
    ckpt_path = f"../weights/{model_name}.ckpt"
    model_config_path = "../configs/model.yaml"

    model = prepare_model(
        model_config_path=model_config_path,
        ckpt_path=ckpt_path,
        max_queries=(
            64 if limit_gpu_mem_usage else 128
        ),  # processes point tracking queries in batch of size max_queries
        precision=precision,
        accelerator=accelerator,
    )

    print("=" * 60)
    print("DAVIS Examples")
    print("Tasks: depth, flow_2d_backward, dyn_mask, track_2d")
    print("Davis provides instance masks, so we can sample and track points on the instance masks")
    print("=" * 60)

    data_name = "davis"
    data_root = "data/davis/v1"
    test_dataset = DavisDataset(
        data_root=data_root,
        crop_size=(16, 224, 224) if limit_gpu_mem_usage else None,
        estimation_directions=[1],
        track_2d_querry_sampling_spacing=0.02,
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    fabric = Fabric(precision=precision, accelerator=accelerator)  # type: ignore
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    data_iter = iter(test_dataloader)

    # set tasks
    tasks = ["depth", "flow_2d_backward", "dyn_mask", "track_2d"]

    # Forward pass and visualize results
    for i in range(min(10, len(test_dataset))):
        batch = next(data_iter)
        print("Running inference for seq: ", batch["seq_name"][0])
        with torch.no_grad():
            out = model.forward(batch, tasks)
        if i == 0:
            print("The model outputs: ", out.keys())
        print("Generating visualization")
        out_path = os.path.join("results", model_name, data_name)
        out_vid, out_vid_name = generate_video_visualizations(batch, out, tasks, out_path)
        print("Result saved to: ", out_vid_name)

    print("=" * 60)
    print("Run on any general video")
    print("Tasks: depth, flow_2d_backward, dyn_mask, track_2d")
    print("=" * 60)
    # You can add your videos to data/videos
    data_name = "videos"
    video_paths = ["data/davis/davis_train.mp4", "data/video/galileo.mp4"]
    test_dataset = VideoDataset(
        video_paths=video_paths,
        crop_size=(16, 224, 224) if limit_gpu_mem_usage else (64, 224, 224),
        estimation_directions=[1],
        track_2d_querry_sampling_spacing=0.04,
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    fabric = Fabric(precision=precision, accelerator=accelerator)  # type: ignore
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    data_iter = iter(test_dataloader)

    # set tasks
    tasks = ["depth", "flow_2d_backward", "dyn_mask", "track_2d"]

    # Forward pass and visualize results
    for i in range(min(10, len(test_dataset))):
        batch = next(data_iter)
        print("Running inference for seq: ", batch["seq_name"][0])
        with torch.no_grad():
            out = model.forward(batch, tasks)
        if i == 0:
            print("The model outputs: ", out.keys())
        print("Generating visualization")
        out_path = os.path.join("results", model_name, data_name)
        out_vid, out_vid_name = generate_video_visualizations(batch, out, tasks, out_path)
        print("Result saved to: ", out_vid_name)

    print("=" * 60)
    print("DAVIS Examples")
    print("Tasks: depth, flow_2d_backward, dyn_mask, track_2d, camray")
    print("DAVIS does not provide intrinsics, so we estimate them as well.")
    print(
        "We can use the estimated cameras to visualize depth, cameras, and 3D tracks in a consistent reference frame."
    )
    print("=" * 60)
    data_name = "davis"
    data_root = "data/davis/v2"
    test_dataset = DavisDataset(
        data_root=data_root,
        crop_size=(16, 224, 224) if limit_gpu_mem_usage else (56, 224, 224),
        estimation_directions=[1],
        track_2d_querry_sampling_spacing=0.02,
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    fabric = Fabric(precision=precision, accelerator=accelerator)  # type: ignore
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    data_iter = iter(test_dataloader)

    # set tasks
    tasks = ["depth", "flow_2d_backward", "dyn_mask", "track_2d", "camray"]

    for i in range(min(10, len(test_dataset))):
        batch = next(data_iter)
        print("Running inference for seq: ", batch["seq_name"][0])
        with torch.no_grad():
            out = model.forward(batch, tasks)
        if i == 0:
            print("The model outputs: ", out.keys())
        print("Generating visualization")
        out_path = os.path.join("results", "4d_recon", model_name, data_name)
        out_vid, out_vid_name = generate_video_visualizations(batch, out, tasks, out_path)
        print("Result saved to: ", out_vid_name)
        ply_files = generate_4D_visualization(batch, out, tasks, out_path)
        if len(ply_files) > 0:
            port = start_port + vis_count  # type: ignore
            vis = visualize_point_cloud_viser(
                ply_files,
                port=port,
                seq_name=batch["seq_name"][0],  # type: ignore
                point_size=0.02,
            )
            vis_list.append(vis)
            vis_count += 1  # type: ignore

    print("=" * 60)
    print("Running on any general video")
    print("Tasks: depth, flow_2d_backward, dyn_mask, track_2d, camray")
    print(
        "We can use the estimated cameras to visualize depth, cameras, and 3D tracks in a consistent reference frame."
    )
    print("=" * 60)
    data_name = "videos"
    video_paths = ["data/davis/davis_train.mp4", "data/video/galileo.mp4"]
    test_dataset = VideoDataset(
        video_paths=video_paths,
        crop_size=(16, 224, 224) if limit_gpu_mem_usage else (64, 224, 224),
        estimation_directions=[1],
        track_2d_querry_sampling_spacing=0.04,
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    fabric = Fabric(precision=precision, accelerator=accelerator)  # type: ignore
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    data_iter = iter(test_dataloader)

    # set tasks
    tasks = ["depth", "flow_2d_backward", "dyn_mask", "track_2d", "camray"]

    for i in range(min(10, len(test_dataset))):
        batch = next(data_iter)
        print("Running inference for seq: ", batch["seq_name"][0])
        with torch.no_grad():
            out = model.forward(batch, tasks)
        if i == 0:
            print("The model outputs: ", out.keys())
        print("Generating visualization")
        out_path = os.path.join("results", "4d_recon", model_name, data_name)
        out_vid, out_vid_name = generate_video_visualizations(batch, out, tasks, out_path)
        print("Result saved to: ", out_vid_name)
        ply_files = generate_4D_visualization(batch, out, tasks, out_path)
        if len(ply_files) > 0:
            port = start_port + vis_count  # type: ignore
            vis = visualize_point_cloud_viser(
                ply_files,
                port=port,
                seq_name=batch["seq_name"][0],  # type: ignore
                point_size=0.02,
            )
            vis_list.append(vis)
            vis_count += 1  # type: ignore

    print("=" * 60)
    print("Running on Dycheck dataset with input camera intrinsics")
    print("Tasks: depth, flow_2d_backward, dyn_mask, track_2d, camray")
    print("We can visualize the estimated cameras and depth in a consistent reference frame.")
    print("=" * 60)
    original_flag_use_intrinsics = model.l4p_model.task_heads["camray"].use_intrinsics
    model.l4p_model.task_heads["camray"].use_intrinsics = True  # uses input camera intrinsics

    data_name = "dycheck"
    data_root = "data/dycheck/extracted"
    test_dataset = DycheckDataset(
        data_root=data_root,
        resize_size=(298, 224),  # resize but keep aspect ratio
        crop_size=(16,224,224) if limit_gpu_mem_usage else (64,224,224),
        stride=2,
        track_2d_querry_sampling_spacing=0.04,
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    fabric = Fabric(precision=precision, accelerator=accelerator)  # type: ignore
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    data_iter = iter(test_dataloader)

    # set tasks
    tasks = ["depth", "flow_2d_backward", "dyn_mask", "track_2d", "camray"]

    for i in range(min(10, len(test_dataset))):
        batch = next(data_iter)
        print("Running inference for seq: ", batch["seq_name"][0])
        with torch.no_grad():
            out = model.forward(batch, tasks)
        if i == 0:
            print("The model outputs: ", out.keys())
        print("Generating visualization")
        out_path = os.path.join("results", "4d_recon", model_name, data_name)
        out_vid, out_vid_name = generate_video_visualizations(batch, out, tasks, out_path)
        print("Result saved to: ", out_vid_name)
        ply_files = generate_4D_visualization(batch, out, tasks, out_path)
        if len(ply_files) > 0:
            port = start_port + vis_count  # type: ignore
            vis = visualize_point_cloud_viser(
                ply_files,
                port=port,
                seq_name=batch["seq_name"][0],  # type: ignore
                point_size=0.02,
            )
            vis_list.append(vis)
            vis_count += 1  # type: ignore

    model.l4p_model.task_heads["camray"].use_intrinsics = original_flag_use_intrinsics

    # sleep
    try:
        # Keep server running
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
