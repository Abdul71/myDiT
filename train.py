import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import wandb
import click
from torchvision import datasets, transforms
from models import DiT_models 
from data import prepare_mnist_data  

def train(model, trainloader, device, optimizer, criterion, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            images, labels = images.to(device), labels.to(device)

            b, c, h, w = images.shape
            t = torch.rand(b, 1, 1, 1, device=device) 
            noises = torch.randn(b, c, h, w, device=device)

            corrupted = t * noises + (1 - t) * images
            target = images - noises

            optimizer.zero_grad()

            outputs = model(corrupted, t.squeeze(), labels)

            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:  
                avg_loss = running_loss / 50
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}', flush=True)
                losses.append(avg_loss)
                wandb.log({"loss": avg_loss})
                running_loss = 0.0

    print('Finished Training', flush=True)
    return losses

def evaluate(model, device, model_save_dir):
    model.eval()
    output_save_dir = os.path.join(model_save_dir, "output")
    os.makedirs(output_save_dir, exist_ok=True)

    b, c, h, w = 1, 1, 32, 32
    random_noise = torch.randn(b, c, h, w, device=device)

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    with torch.no_grad():
        for n in range(10):  # Iterate over each label
            output = random_noise
            for i in range(100):
                t = torch.tensor([i / 100], device=device)
                t = 1-t
                output = output + model(output, t, torch.tensor([n], device=device)) * 1 / 100

            output_image = output.cpu().squeeze().detach().numpy()
            axs[n].imshow(output_image, cmap='gray')
            axs[n].set_title(f'Label {n}')
            axs[n].axis('off')

    plt.tight_layout()
    output_plot_path = os.path.join(output_save_dir, 'consolidated_output.png')
    plt.savefig(output_plot_path)
    print(f'Consolidated output plot saved to {output_plot_path}', flush=True)
    wandb.log({"consolidated_output": wandb.Image(output_plot_path)})
    plt.close()
        

@click.command()
@click.option('--batch_size', default=256, help='Batch size for training')
@click.option('--num_workers', default=4, help='Number of data loader workers')
@click.option('--model_name', default='DiT-L/4', help='Model name from DiT_models')
@click.option('--input_size', default=32, help='Input image size')
@click.option('--in_channels', default=1, help='Number of input channels')
@click.option('--num_classes', default=10, help='Number of output classes')
@click.option('--learn_sigma', is_flag=True, help='Flag to learn sigma')
@click.option('--learning_rate', default=3e-4, help='Learning rate for optimizer')
@click.option('--num_epochs', default=256, help='Number of training epochs')
def main(batch_size, num_workers, model_name, input_size, in_channels, num_classes, learn_sigma, learning_rate, num_epochs):

    # Directory setup
    model_folder_name = f"model_{model_name.replace('/', '_')}_epochs_{num_epochs}_lr_{learning_rate}_bs_{batch_size}"
    model_save_dir = os.path.join('models', model_folder_name)
    os.makedirs(model_save_dir, exist_ok=True)

    model_save_path = os.path.join(model_save_dir, "model.pth")
    plot_save_path = os.path.join(model_save_dir, "training_loss_plot.png")

    # Initialize Weights and Biases (wandb)
    wandb.init(project="Diffuse", entity="abdo71", config={
        "architecture": "DiT",
        "Dataset": "MNIST",
        "batch_size": batch_size,
        "num_workers": num_workers,
        "model_name": model_name,
        "input_size": input_size,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "learn_sigma": learn_sigma,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs
    }, name=f"run_{model_folder_name}")

    print("Starting run with the following configuration:")
    print(f"batch_size: {batch_size}")
    print(f"num_workers: {num_workers}")
    print(f"model_name: {model_name}")
    print(f"input_size: {input_size}")
    print(f"in_channels: {in_channels}")
    print(f"num_classes: {num_classes}")
    print(f"learn_sigma: {learn_sigma}")
    print(f"learning_rate: {learning_rate}")
    print(f"num_epochs: {num_epochs}")

    # Data Preparation
    trainloader, testloader, classes = prepare_mnist_data(batch_size=batch_size, num_workers=num_workers)

    # Device setup
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Model, Loss, Optimizer
    model = DiT_models[model_name](input_size=input_size, in_channels=in_channels, num_classes=num_classes, learn_sigma=learn_sigma).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    losses = train(model, trainloader, device, optimizer, criterion, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}', flush=True)

    # Plot the losses
    plt.plot(losses)
    plt.xlabel('Iterations (per 50 batches)')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(plot_save_path)
    print(f'Loss plot saved to {plot_save_path}', flush=True)
    wandb.log({"training_loss_plot": wandb.Image(plot_save_path)})
    plt.close()

    # Evaluate the model
    evaluate(model, device, model_save_dir)

if __name__ == "__main__":
    main()
