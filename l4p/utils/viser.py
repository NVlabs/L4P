# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import numpy as np
import open3d as o3d
import trimesh
import viser


def visualize_point_cloud_viser(
    file_paths,
    port: int = 8006,
    seq_name: str = "point cloud",
    point_size: float = 0.02,
    dark_mode: bool = True,
):
    # read point clouds
    pcds = []
    meshes = []
    names = []
    for filepath in file_paths:
        # Loads all the point clouds and meshes
        pcds_curr = []
        meshes_curr = []
        for key in filepath.keys():
            if "pc" in key:
                filename = os.path.basename(filepath[key])
                print(f"  Loading: {filename} ...", end="")
                pcds_curr.append(o3d.io.read_point_cloud(filepath[key]))
            if "mesh" in key:
                filename = os.path.basename(filepath[key])
                print(f"  Loading: {filename} ...", end="")
                meshes_curr.append(trimesh.load(filepath[key]))
        pcds.append(pcds_curr)
        meshes.append(meshes_curr)
        names.append(filepath["name"])
    print(f"Successfully loaded {len(pcds)} point clouds and {len(meshes)} meshes.")

    vis = viser.ViserServer(port=port, label=seq_name)
    vis.gui.configure_theme(dark_mode=dark_mode)

    slider = vis.gui.add_slider(
        "point cloud id",
        min=0,
        max=len(pcds) - 1,
        step=1,
        initial_value=0,
    )

    # Add a text display for the current filename
    filename_text = vis.gui.add_text("Current File", initial_value="N/A", disabled=True)

    # Update Logic
    def update_displayed_point_cloud():
        current_index = slider.value
        filename = names[current_index]
        for i in range(len(pcds[current_index])):
            points = np.asarray(pcds[current_index][i].points)
            colors = np.asarray(pcds[current_index][i].colors)

            # Add/replace the point cloud in Viser
            vis.scene.add_point_cloud(
                name=f"/{seq_name}_pc_{i}",
                points=points,
                colors=colors,
                point_size=point_size,
            )

        for i in range(len(meshes[current_index])):
            mesh = meshes[current_index][i]
            vis.scene.add_mesh_trimesh(name=f"/{seq_name}_mesh_{i}", mesh=mesh)

        # Update the filename display
        filename_text.value = os.path.basename(filename)

    # Connect Slider
    slider.on_update(lambda _: update_displayed_point_cloud())

    # Initial Display
    # Call update once to show the first point cloud
    update_displayed_point_cloud()

    print("\nServer running. Open your browser to the address provided by Viser.")
    print("Use the slider in the GUI to switch between point clouds.")
    return vis
