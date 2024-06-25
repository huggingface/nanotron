import os

import gradio as gr
import numpy as np
import torch


def load_folder_structure(root_dir):
    folder_structure = {}
    for step_dir in sorted(os.listdir(root_dir), key=lambda x: int(x)):
        step = int(step_dir)
        step_path = os.path.join(root_dir, step_dir)
        folder_structure[step] = []
        for folder in os.listdir(step_path):
            if os.path.isdir(os.path.join(step_path, folder)):
                folder_structure[step].append(folder)
    return folder_structure


def load_tensors(root_dir, selected_steps, selected_folders):
    tensors = {}
    for step in selected_steps:
        step_dir = os.path.join(root_dir, str(step))
        for folder in selected_folders:
            folder_path = os.path.join(step_dir, folder)
            if os.path.isdir(folder_path):
                for tensor_file in os.listdir(folder_path):
                    tensor_name = os.path.splitext(tensor_file)[0]
                    tensor_path = os.path.join(folder_path, tensor_file)

                    # Load the tensor
                    tensor = torch.load(tensor_path)

                    # Calculate mean and standard deviation
                    if not isinstance(tensor, torch.Tensor):
                        continue

                    mean = tensor.mean().item()
                    std = tensor.std().item()

                    if tensor_name not in tensors:
                        tensors[tensor_name] = {"steps": [], "means": [], "stds": []}

                    tensors[tensor_name]["steps"].append(step)
                    tensors[tensor_name]["means"].append(mean)
                    tensors[tensor_name]["stds"].append(std)
    return tensors


def create_gradio_interface():
    def update_steps(root_dir):
        folder_structure = load_folder_structure(root_dir)
        steps = list(folder_structure.keys())
        return gr.Dropdown(choices=steps, multiselect=True, value=steps)

    def update_folders(root_dir, selected_steps):
        folder_structure = load_folder_structure(root_dir)
        folders = set()
        for step in selected_steps:
            folders.update(folder_structure[step])
        return gr.Dropdown(choices=list(folders), multiselect=True, value=list(folders))

    def update_tensor_names(root_dir, selected_steps, selected_folders):
        tensors = load_tensors(root_dir, selected_steps, selected_folders)
        tensor_names = list(tensors.keys())
        default_tensor = np.random.choice(tensor_names) if tensor_names else None
        return gr.Dropdown(choices=tensor_names, multiselect=True, value=[default_tensor] if default_tensor else [])

    def update_line_plot(root_dir, selected_steps, selected_folders, selected_tensors, show_std):
        tensors = load_tensors(root_dir, selected_steps, selected_folders)

        fig = go.Figure()

        all_means = []
        for tensor_name in selected_tensors:
            if tensor_name in tensors:
                data = tensors[tensor_name]
                steps = data["steps"]
                means = data["means"]
                stds = data["stds"]
                all_means.extend(means)

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=means,
                        mode="lines+markers",
                        name=tensor_name,
                        line={"width": 2},
                        hovertemplate="Step: %{x}<br>Mean: %{y:.4f}<br>Std: %{text:.4f}<extra></extra>",
                        text=stds,
                    )
                )

                if show_std:
                    fig.add_trace(
                        go.Scatter(
                            x=steps + steps[::-1],
                            y=[m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means[::-1], stds[::-1])],
                            fill="toself",
                            fillcolor="rgba(0,100,80,0.2)",
                            line={"color": "rgba(255,255,255,0)"},
                            hoverinfo="skip",
                            showlegend=False,
                            name=tensor_name + " Std Dev",
                        )
                    )

        fig.update_layout(
            title="Tensor Means and Variances Across Training Steps",
            xaxis_title="Training Step",
            yaxis_title="Mean of Tensors",
            hovermode="x unified",
            legend_title="Tensor Names",
        )

        # Adjust y-axis range to focus on the trend
        if all_means:
            y_mean = np.mean(all_means)
            y_std = np.std(all_means)
            y_range = [y_mean - 3 * y_std, y_mean + 3 * y_std]
            fig.update_yaxes(range=y_range)

        return fig

    import plotly.graph_objects as go

    def update_plots(root_dir, selected_steps, selected_folders, selected_tensors, exp_min, exp_max):
        if not selected_tensors:
            return None, None, "No tensors selected", None

        tensors = load_tensors(root_dir, selected_steps, selected_folders)

        all_values = {}
        for step in selected_steps:
            all_values[step] = []
            for tensor_name in selected_tensors:
                if tensor_name in tensors:
                    step_dir = os.path.join(root_dir, str(step))
                    for folder in selected_folders:
                        tensor_path = os.path.join(step_dir, folder, f"{tensor_name}.pt")
                        if os.path.exists(tensor_path):
                            tensor = torch.load(tensor_path)
                            all_values[step].extend(tensor.flatten().tolist())

        if not all_values:
            return None, None, "No data available", None

        # Create histogram with slider
        hist_fig = go.Figure()

        max_hist_value = 0
        for step in selected_steps:
            values = np.array(all_values[step])
            hist, bin_edges = np.histogram(values, bins=50)
            hist_fig.add_trace(go.Bar(x=bin_edges[:-1], y=hist, name=f"Step {step}", visible=False))
            max_hist_value = max(max_hist_value, np.max(hist))

        # Make the first trace visible
        hist_fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i, step in enumerate(selected_steps):
            step = {
                "method": "update",
                "args": [
                    {"visible": [False] * len(hist_fig.data)},
                    {"title": f"Distribution of Tensor Values (Step {step})"},
                ],
                "label": str(step),
            }
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [{"active": 0, "currentvalue": {"prefix": "Step: "}, "pad": {"t": 50}, "steps": steps}]

        hist_fig.update_layout(
            sliders=sliders,
            xaxis_title="Tensor Values",
            yaxis_title="Frequency",
            bargap=0.1,
        )

        # Auto-zoom for histogram
        all_values_flat = np.concatenate(list(all_values.values()))
        q1, q3 = np.percentile(all_values_flat, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        hist_fig.update_yaxes(range=[0, max_hist_value * 1.1])
        hist_fig.update_xaxes(range=[lower_bound, upper_bound])

        # Create scatter plot (unchanged)
        scatter_fig = go.Figure()
        for step, values in all_values.items():
            scatter_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(values)),
                    y=values,
                    mode="markers",
                    marker={"size": 2, "opacity": 0.5},
                    name=f"Step {step}",
                )
            )
        scatter_fig.update_layout(
            title="Scatter Plot of Tensor Values",
            xaxis_title="Index",
            yaxis_title="Tensor Value",
        )
        scatter_fig.update_yaxes(range=[lower_bound, upper_bound])

        # Calculate percentages for the first step (you might want to adjust this)
        first_step = selected_steps[0]
        current_values = np.array(all_values[first_step])
        total = len(current_values)
        within_range = np.sum((current_values >= exp_min) & (current_values <= exp_max))
        below_min = np.sum(current_values < exp_min)
        above_max = np.sum(current_values > exp_max)

        percentage_text = f"""
        Step {first_step}:
        Within range [{exp_min}, {exp_max}]: {within_range/total*100:.2f}%
        Below {exp_min}: {below_min/total*100:.2f}%
        Above {exp_max}: {above_max/total*100:.2f}%
        """

        # Create distribution table data for the first step
        hist, bin_edges = np.histogram(current_values, bins=50)
        table_data = []
        for i in range(len(hist)):
            range_start = bin_edges[i]
            range_end = bin_edges[i + 1]
            count = hist[i]
            percentage = count / total * 100
            table_data.append([f"{range_start:.4f} - {range_end:.4f}", f"{percentage:.2f}%", f"{count}"])

        return hist_fig, scatter_fig, percentage_text, table_data

    with gr.Blocks() as iface:
        gr.Markdown("# Neural Networks Debugging Tool")

        with gr.Row():
            root_dir_input = gr.Textbox(label="Root Directory")

        with gr.Row():
            step_dropdown = gr.Dropdown(multiselect=True, label="Select Training Steps")

        with gr.Row():
            folder_dropdown = gr.Dropdown(multiselect=True, label="Select Folders")

        with gr.Row():
            tensor_dropdown = gr.Dropdown(multiselect=True, label="Select Tensors to Display")

        with gr.Row():
            show_std = gr.Checkbox(label="Show Standard Deviation", value=True)
            update_line_plot_button = gr.Button("Update Line Plot")

        with gr.Row():
            line_plot_output = gr.Plot()

        with gr.Row():
            with gr.Column(scale=1):
                exp_min = gr.Number(label="Expected Minimum")
                exp_max = gr.Number(label="Expected Maximum")
                percentage_output = gr.Textbox(label="Value Distribution")
                update_plots_button = gr.Button("Update Plots")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Distribution of Tensor Values"):
                        histogram_output = gr.Plot()
                    with gr.TabItem("Scatter Plot of Tensor Values"):
                        scatter_output = gr.Plot()

        with gr.Row():
            distribution_table = gr.Dataframe(headers=["Range", "Percentage", "Count"], label="Distribution Table")

        root_dir_input.change(fn=update_steps, inputs=[root_dir_input], outputs=[step_dropdown])

        step_dropdown.change(fn=update_folders, inputs=[root_dir_input, step_dropdown], outputs=[folder_dropdown])

        folder_dropdown.change(
            fn=update_tensor_names, inputs=[root_dir_input, step_dropdown, folder_dropdown], outputs=[tensor_dropdown]
        )

        update_line_plot_button.click(
            fn=update_line_plot,
            inputs=[root_dir_input, step_dropdown, folder_dropdown, tensor_dropdown, show_std],
            outputs=[line_plot_output],
        )

        # Add this new event handler
        show_std.change(
            fn=update_line_plot,
            inputs=[root_dir_input, step_dropdown, folder_dropdown, tensor_dropdown, show_std],
            outputs=[line_plot_output],
        )

        update_plots_button.click(
            fn=update_plots,
            inputs=[root_dir_input, step_dropdown, folder_dropdown, tensor_dropdown, exp_min, exp_max],
            outputs=[histogram_output, scatter_output, percentage_output, distribution_table],
        )

    return iface


# Usage
iface = create_gradio_interface()
iface.launch(share=True, debug=True)
