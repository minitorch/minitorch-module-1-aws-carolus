"""
Script to train scalar models on all datasets and save the results.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import minitorch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import minitorch
from project.run_scalar import ScalarTrain

# Configuration
PTS = 50
HIDDEN = 10
RATE = 0.5
MAX_EPOCHS = 500

# Datasets to train on
DATASETS = ["Simple", "Diag", "Xor", "Circle"]

# Directory to save results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_dataset(dataset_name):
    """Train a model on the specified dataset and save the results."""
    print(f"Training on {dataset_name} dataset...")
    
    # Create dataset
    dataset = minitorch.datasets[dataset_name](PTS)
    
    # Create a custom log function to capture the training progress
    training_log = []
    
    def log_fn(epoch, total_loss, correct, losses):
        training_log.append((epoch, total_loss, correct))
        if epoch % 50 == 0 or epoch == MAX_EPOCHS:
            print(f"Epoch {epoch}, Loss: {total_loss:.6f}, Correct: {correct}/{dataset.N}")
    
    # Train the model
    trainer = ScalarTrain(HIDDEN)
    trainer.train(dataset, RATE, MAX_EPOCHS, log_fn)
    
    # Save the training log
    log_file = os.path.join(RESULTS_DIR, f"{dataset_name}_training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Points: {PTS}, Hidden Layers: {HIDDEN}, Learning Rate: {RATE}\n")
        f.write("Epoch, Loss, Correct\n")
        for epoch, loss, correct in training_log:
            f.write(f"{epoch}, {loss:.6f}, {correct}\n")
    
    # Create a visualization of the decision boundary
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a grid of points
    x_range = np.linspace(0, 1, 100)
    y_range = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Evaluate the model on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_1 = X[i, j]
            x_2 = Y[i, j]
            z = trainer.run_one((x_1, x_2))
            Z[i, j] = z.data > 0.5
    
    # Plot the decision boundary
    ax.contourf(X, Y, Z, alpha=0.5, cmap='RdBu')
    
    # Plot the training data
    for i in range(dataset.N):
        x_1, x_2 = dataset.X[i]
        y = dataset.y[i]
        color = "blue" if y == 1 else "red"
        ax.scatter(x_1, x_2, color=color)
    
    ax.set_title(f"{dataset_name} Dataset - Decision Boundary")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    
    # Save the figure
    fig_file = os.path.join(RESULTS_DIR, f"{dataset_name}_decision_boundary.png")
    plt.savefig(fig_file)
    plt.close(fig)
    
    print(f"Results saved to {log_file} and {fig_file}")
    return log_file, fig_file


def main():
    """Train models on all datasets and update the README."""
    print("Training scalar models on all datasets...")
    
    results = {}
    for dataset_name in DATASETS:
        log_file, fig_file = train_dataset(dataset_name)
        results[dataset_name] = (log_file, fig_file)
    
    print("All training complete!")
    
    # Update the README with the results
    readme_file = "README.md"
    with open(readme_file, "r") as f:
        readme_content = f.read()
    
    # Add the training results section if it doesn't exist
    if "## Training Results" not in readme_content:
        readme_content += "\n\n## Training Results\n\n"
        readme_content += "Results of training scalar models on different datasets:\n\n"
        
        for dataset_name in DATASETS:
            log_file, fig_file = results[dataset_name]
            
            # Read the training log
            with open(log_file, "r") as f:
                log_content = f.readlines()
            
            # Extract the final results
            final_epoch = log_content[-1].split(",")[0]
            final_loss = log_content[-1].split(",")[1].strip()
            final_correct = log_content[-1].split(",")[2].strip()
            
            readme_content += f"### {dataset_name} Dataset\n\n"
            readme_content += f"- Points: {PTS}\n"
            readme_content += f"- Hidden Layers: {HIDDEN}\n"
            readme_content += f"- Learning Rate: {RATE}\n"
            readme_content += f"- Final Loss: {final_loss}\n"
            readme_content += f"- Final Accuracy: {final_correct}/{PTS}\n\n"
            readme_content += f"![{dataset_name} Decision Boundary](results/{dataset_name}_decision_boundary.png)\n\n"
            readme_content += "Training Log:\n```\n"
            readme_content += "".join(log_content[:10])  # Show first 10 lines
            readme_content += "...\n"
            readme_content += "".join(log_content[-5:])  # Show last 5 lines
            readme_content += "```\n\n"
        
        with open(readme_file, "w") as f:
            f.write(readme_content)
        
        print(f"README updated with training results!")


if __name__ == "__main__":
    main()
