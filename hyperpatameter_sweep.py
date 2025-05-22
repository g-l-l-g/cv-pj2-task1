import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import shutil
import torch
from datetime import datetime

import config_train as train_cfg
from main_finetune import main_finetune
from model import models
from caltech101_dataprep import config as dp_config
from caltech101_dataprep import data_utils as dp_utils

# --- 定义要尝试的超参数组合 ---
experiment_configurations = [
    # A. Epoch Sweep (Adam, lr_base=1e-4, lr_fc=1e-3, bs=32, scheduler=True)
    {
        "id": "ep1_adam_lr1e4_1e3_bs32", "type": "finetune", "model": "resnet18", "epochs": 1,
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "ep5_adam_lr1e4_1e3_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,  # 参考配置
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "ep10_adam_lr1e4_1e3_bs32", "type": "finetune", "model": "resnet18", "epochs": 10,
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "ep15_adam_lr1e4_1e3_bs32", "type": "finetune", "model": "resnet18", "epochs": 15,
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },

    # B. Learning Rate Sweep (Adam, epochs=5, bs=32, scheduler=True)
    # lr_base=1e-4, lr_fc=1e-3 的参考配置是 ep5_adam_lr1e4_1e3_bs32
    {
        "id": "lr_b1e5_f1e4_adam_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 1e-5, "lr_fc": 1e-4, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "lr_b5e5_f5e4_adam_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 5e-5, "lr_fc": 5e-4, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "lr_b2e4_f2e3_adam_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 2e-4, "lr_fc": 2e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "lr_b5e4_f5e3_adam_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 5e-4, "lr_fc": 5e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "lr_b1e4_f1e4_adam_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,  # 基准层LR == 全连接层LR (较低)
        "lr_base": 1e-4, "lr_fc": 1e-4, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "lr_b1e3_f1e3_adam_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,  # 基准层LR == 全连接层LR (较高)
        "lr_base": 1e-3, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },

    # C. Optimizer Sweep (epochs=5, bs=32, scheduler=True)
    # Adam 参考配置是 ep5_adam_lr1e4_1e3_bs32 (LR base=1e-4, fc=1e-3)
    {
        "id": "opt_sgd_lr1e3_1e2_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 1e-3, "lr_fc": 1e-2, "optimizer": "SGD", "wd": 1e-4, "batch_size": 32,  # SGD 常用学习率
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "opt_sgd_lr5e4_5e3_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 5e-4, "lr_fc": 5e-3, "optimizer": "SGD", "wd": 1e-4, "batch_size": 32,  # SGD 中等学习率
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "opt_sgd_lr1e2_1e1_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,  # SGD 较高学习率
        "lr_base": 1e-2, "lr_fc": 1e-1, "optimizer": "SGD", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },

    # D. Batch Size Sweep (Adam, epochs=5, lr_base=1e-4, lr_fc=1e-3, scheduler=True)
    # BS=32 参考配置是 ep5_adam_lr1e4_1e3_bs32
    {
        "id": "bs8_adam_lr1e4_1e3_e5", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 8,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "bs64_adam_lr1e4_1e3_e5", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 64,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    {
        "id": "bs128_adam_lr1e4_1e3_e5", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 128,
        "use_scheduler": True, "scheduler_patience": 3, "scheduler_factor": 0.1
    },
    # E. Scheduler Variation (Adam, epochs=5, lr_base=1e-4, lr_fc=1e-3, bs=32)
    {
        "id": "no_sched_adam_lr1e4_1e3_e5_bs32", "type": "finetune", "model": "resnet18", "epochs": 5,
        "lr_base": 1e-4, "lr_fc": 1e-3, "optimizer": "Adam", "wd": 1e-4, "batch_size": 32,
        "use_scheduler": False, "scheduler_patience": 3, "scheduler_factor": 0.1  # scheduler_patience/factor 无效
    }
]


def cleanup_experiment_artifacts(exp_id, model_name, train_type, sweep_root_dir):
    """Deletes checkpoints and TensorBoard logs for a specific experiment."""
    experiment_log_suffix = f"{train_type}_{model_name}_{exp_id}"

    current_checkpoint_dir = os.path.join(sweep_root_dir, "checkpoints", experiment_log_suffix)

    base_tb_log_dir_for_exp_type = os.path.join(sweep_root_dir, "runs", f"{train_type}_{model_name}")
    actual_tb_log_dir_for_exp = os.path.join(base_tb_log_dir_for_exp_type, experiment_log_suffix)

    if os.path.exists(current_checkpoint_dir):
        print(f"  Cleaning up old checkpoint directory: {current_checkpoint_dir}")
        shutil.rmtree(current_checkpoint_dir)

    if os.path.exists(actual_tb_log_dir_for_exp):
        print(f"  Cleaning up old TensorBoard log directory: {actual_tb_log_dir_for_exp}")
        shutil.rmtree(actual_tb_log_dir_for_exp)


if __name__ == '__main__':
    results = []
    # All outputs will be under this single root directory
    main_sweep_dir_name = "hyperparams_search"
    os.makedirs(main_sweep_dir_name, exist_ok=True)

    # Create subdirectories if they might not be created by underlying functions
    os.makedirs(os.path.join(main_sweep_dir_name, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(main_sweep_dir_name, "runs"), exist_ok=True)
    os.makedirs(os.path.join(main_sweep_dir_name, "visualizations"), exist_ok=True)

    # --- 执行实验 ---
    for i, config in enumerate(experiment_configurations):
        print(f"\n--- Starting Experiment {i + 1}/{len(experiment_configurations)}: ID = {config['id']} ---")
        start_run_time = time.time()

        # 1. 更新全局配置 (config_train.py 和 dp_config.py)
        train_cfg.MODEL_NAME = config["model"]
        train_cfg.EPOCHS = config["epochs"]
        train_cfg.BATCH_SIZE = config["batch_size"]
        dp_config.BATCH_SIZE = config["batch_size"]

        train_cfg.USE_LR_SCHEDULER = config["use_scheduler"]
        train_cfg.LR_SCHEDULER_PATIENCE = config["scheduler_patience"]
        train_cfg.LR_SCHEDULER_FACTOR = config["scheduler_factor"]

        log_suffix = f"{config['type']}_{config['model']}_{config['id']}"

        train_cfg.FINETUNE_LEARNING_RATE_BASE = config["lr_base"]
        train_cfg.FINETUNE_LEARNING_RATE_FC = config["lr_fc"]
        train_cfg.FINETUNE_OPTIMIZER = config["optimizer"]
        train_cfg.FINETUNE_WEIGHT_DECAY = config["wd"]

        # Checkpoints will be under hyperparams_search/checkpoints/experiment_suffix
        run_specific_checkpoint_dir = os.path.join(main_sweep_dir_name, "checkpoints", log_suffix)
        train_cfg.CHECKPOINT_DIR_FINETUNE = run_specific_checkpoint_dir
        # Ensure this specific checkpoint directory is created before training, main_finetune might also do this.
        os.makedirs(train_cfg.CHECKPOINT_DIR_FINETUNE, exist_ok=True)

        base_log_dir_for_tb = os.path.join(main_sweep_dir_name, "runs", f"{config['type']}_{config['model']}")
        train_cfg.LOG_DIR_FINETUNE = base_log_dir_for_tb
        train_cfg.LOG_DIR_FINETUNE_EXPERIMENT_NAME = log_suffix

        os.makedirs(os.path.join(train_cfg.LOG_DIR_FINETUNE, train_cfg.LOG_DIR_FINETUNE_EXPERIMENT_NAME), exist_ok=True)

        if models.CUSTOM_MODEL_CACHE_DIR:
            os.makedirs(models.CUSTOM_MODEL_CACHE_DIR, exist_ok=True)

        print(
            f"  Config: Epochs={config['epochs']}, LR_base={config['lr_base']}, "
            f"LR_fc={config['lr_fc']}, Optim={config['optimizer']}, BS={config['batch_size']}")
        print(f"  Checkpoint dir: {train_cfg.CHECKPOINT_DIR_FINETUNE}")
        print(
            f"  TensorBoard log base: {train_cfg.LOG_DIR_FINETUNE}, "
            f"Experiment name for TB: {train_cfg.LOG_DIR_FINETUNE_EXPERIMENT_NAME}")

        best_val_acc, test_acc = main_finetune()

        run_duration = time.time() - start_run_time

        best_val_acc_item = best_val_acc.item() if isinstance(best_val_acc, torch.Tensor) else float(best_val_acc)
        test_acc_item = test_acc.item() if isinstance(test_acc, torch.Tensor) else float(test_acc)

        print(f"--- Experiment {config['id']} complete. Duration: {run_duration:.2f} sec ---")
        print(f"    Best val accuracy: {best_val_acc_item:.4f}, Test accuracy: {test_acc_item:.4f}")

        current_result = config.copy()
        current_result["best_val_accuracy"] = round(best_val_acc_item, 4)
        current_result["test_accuracy"] = round(test_acc_item, 4)
        current_result["duration_seconds"] = round(run_duration, 2)
        results.append(current_result)

        results_df_intermediate = pd.DataFrame(results)
        intermediate_csv_path = os.path.join(main_sweep_dir_name, "hyperparameter_sweep_finetune_results_live.csv")
        results_df_intermediate.to_csv(intermediate_csv_path, index=False)
        print(f"Intermediate results saved to: {intermediate_csv_path}")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No experiment results to analyze.")
        sys.exit()

    final_results_filename = os.path.join(main_sweep_dir_name, f"hyperparameter_sweep_finetune_results_final.csv")
    results_df.to_csv(final_results_filename, index=False)
    print(f"\nAll experiment results saved to: {final_results_filename}")

    if "test_accuracy" in results_df.columns and not results_df["test_accuracy"].empty:
        results_df_sorted = results_df.sort_values(by=["test_accuracy", "best_val_accuracy"], ascending=[False, False])
        if not results_df_sorted.empty:
            best_run = results_df_sorted.iloc[0]
            print("\n--- Best Configuration (based on highest test accuracy, then validation accuracy) ---")
            print(best_run)
            print("------------------------------------------------------------------------------------")
        else:
            print("\nWarning: No data after sorting. Cannot determine best configuration.")
    else:
        print("\nWarning: 'test_accuracy' column is missing or empty. Cannot determine best configuration.")

    sns.set_theme(style="whitegrid")
    output_viz_dir = os.path.join(main_sweep_dir_name, "visualizations")

    print(f"Visualizations will be saved to: {output_viz_dir}")

    if "test_accuracy" in results_df.columns and "id" in results_df.columns and not results_df.empty:
        plt.figure(figsize=(14, 8))
        # Use sorted data for consistent bar chart ordering if results_df_sorted exists
        df_for_plot = results_df_sorted if 'results_df_sorted' in locals() and not results_df_sorted.empty else results_df.sort_values(
            by="test_accuracy", ascending=False)
        sns.barplot(data=df_for_plot, x="id", y="test_accuracy", palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("Test Accuracy for Different Hyperparameter Combinations (ResNet18 Finetune)")
        plt.ylabel("Test Accuracy")
        plt.xlabel("Experiment ID")
        plt.tight_layout()
        plt.savefig(os.path.join(output_viz_dir, "test_accuracy_all_runs.png"))
        plt.close()
    else:
        print("Could not generate 'test_accuracy_all_runs.png', missing necessary columns or no data.")

    # For plot b: Impact of Learning Rate
    # Filter for experiments with 5 epochs, Adam optimizer, and batch_size 32
    lr_fc_df = results_df[
        (results_df["epochs"] == 5) &  # MODIFIED: Was 1, now 5
        (results_df["optimizer"] == "Adam") &
        (results_df["batch_size"] == 32)
        ].copy()
    if not lr_fc_df.empty and "lr_fc" in lr_fc_df.columns and "test_accuracy" in lr_fc_df.columns:
        lr_fc_df_sorted = lr_fc_df.sort_values(by="lr_fc")
        plt.figure(figsize=(10, 6))
        lr_fc_df_sorted['lr_labels'] = lr_fc_df_sorted.apply(lambda row: f"fc:{row['lr_fc']}\nbase:{row['lr_base']}",
                                                             axis=1)
        sns.lineplot(data=lr_fc_df_sorted, x="lr_labels", y="test_accuracy", marker='o', sort=False)
        num_epochs_for_title_b = lr_fc_df_sorted['epochs'].iloc[0] if not lr_fc_df_sorted.empty else "N/A" # Will be 5
        plt.title(
            f"Impact of Learning Rate on Test Accuracy (ResNet18 Finetune, Adam, {num_epochs_for_title_b} Epochs)")
        plt.xlabel("Learning Rate Combination (FC Layer / Base Layers)")
        plt.ylabel("Test Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_viz_dir, "lr_impact_on_accuracy.png"))
        plt.close()
    else:
        print("Could not generate 'lr_impact_on_accuracy.png',"
              "insufficient data or missing columns (check epoch filter: 5 epochs, Adam, BS 32).")

    # For plot c: Impact of Epochs
    # Filter for specific LR, optimizer, and batch_size
    epochs_df = results_df[
        (results_df["lr_base"] == 1e-4) &
        (results_df["lr_fc"] == 1e-3) &
        (results_df["optimizer"] == "Adam") &
        (results_df["batch_size"] == 32)
        ].copy()
    if not epochs_df.empty and "epochs" in epochs_df.columns and "test_accuracy" in epochs_df.columns:
        epochs_df_sorted = epochs_df.sort_values(by="epochs")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=epochs_df_sorted, x="epochs", y="test_accuracy", marker='o', label="Test Accuracy")
        sns.lineplot(data=epochs_df_sorted, x="epochs", y="best_val_accuracy", marker='x', linestyle='--',
                     label="Best Validation Accuracy")
        plt.title("Impact of Epochs on Accuracy (ResNet18 Finetune, Adam, LR_fc=1e-3, LR_base=1e-4, BS=32)")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Accuracy")
        if not epochs_df_sorted["epochs"].empty and len(
                epochs_df_sorted["epochs"].unique()) > 1:  # Ensure multiple unique epochs for sensible xticks
            plt.xticks(sorted(epochs_df_sorted["epochs"].unique()))
        elif not epochs_df_sorted["epochs"].empty:  # Single epoch value
            plt.xticks([epochs_df_sorted["epochs"].iloc[0]])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_viz_dir, "epochs_impact_on_accuracy.png"))
        plt.close()
    else:
        print("Could not generate 'epochs_impact_on_accuracy.png', insufficient data or missing columns.")

    # For plot d: Impact of Optimizer
    # Filter for experiments with 5 epochs and batch_size 32
    optimizer_df = results_df[
        (results_df["epochs"] == 5) &  # MODIFIED: Was 1, now 5
        (results_df["batch_size"] == 32)
        ].copy()
    if not optimizer_df.empty and "optimizer" in optimizer_df.columns and "test_accuracy" in optimizer_df.columns:
        plt.figure(figsize=(10, 6))
        # To make x-axis more readable, combine optimizer and LRs for SGD runs in the ID if not already clear
        # Or ensure IDs are distinct enough. Current IDs for optimizers are distinct.
        sns.barplot(data=optimizer_df, x="id", y="test_accuracy", hue="optimizer", dodge=False)
        plt.xticks(rotation=25, ha="right")
        plt.title("Test Accuracy for Different Optimizers (ResNet18 Finetune, 5 Epochs, BS=32)")  # MODIFIED title
        plt.ylabel("Test Accuracy")
        plt.xlabel("Experiment ID (Implies Optimizer and Learning Rate)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_viz_dir, "optimizer_comparison.png"))
        plt.close()
    else:
        print(
            "Could not generate 'optimizer_comparison.png', insufficient data or missing columns (check epoch filter: 5 epochs, BS 32).")

    # For plot e: Impact of Batch Size
    # Filter for experiments with 5 epochs, specific LR, and Adam optimizer
    bs_df = results_df[
        (results_df["epochs"] == 5) &  # MODIFIED: Was 1, now 5
        (results_df["lr_base"] == 1e-4) &
        (results_df["lr_fc"] == 1e-3) &
        (results_df["optimizer"] == "Adam")
        ].copy()
    if not bs_df.empty and "batch_size" in bs_df.columns and "test_accuracy" in bs_df.columns:
        bs_df_sorted = bs_df.sort_values(by="batch_size")
        plt.figure(figsize=(8, 6))
        sns.barplot(data=bs_df_sorted, x="batch_size", y="test_accuracy")
        plt.title(
            "Impact of Batch Size on Test Accuracy (ResNet18 Finetune, Adam, 5 Epochs, Default LR)")  # MODIFIED title
        plt.xlabel("Batch Size")
        plt.ylabel("Test Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(output_viz_dir, "batch_size_impact.png"))
        plt.close()
    else:
        print("Could not generate 'batch_size_impact.png', insufficient data or missing columns (check epoch filter: 5 epochs, Adam, specific LR).")

    print(f"\nHyperparameter sweep complete. Results and charts saved in '{main_sweep_dir_name}' directory.")
    print(
        f"Use TensorBoard to view detailed training progress: tensorboard --logdir "
        f"\"{os.path.join(main_sweep_dir_name, 'runs')}\"")
