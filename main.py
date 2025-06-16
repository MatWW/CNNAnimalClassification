import os
import torch
import argparse
from src.data_pipeline import DataPipeline
from src.model import AnimalCNN
from src.train_pipeline import TrainPipeline
from src.eval_pipeline import EvalPipeline


def main():
    parser = argparse.ArgumentParser(description='Animals-10 Classification')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both',
                        help='Mode: train, eval, or both')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to save/load model')

    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_pipeline = DataPipeline(batch_size=args.batch_size)
    model = AnimalCNN().to(device)

    if args.mode in ['train', 'both']:
        print("\nTRAINING")
        train_loader, val_loader, _ = data_pipeline.create_dataloaders()

        trainer = TrainPipeline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.lr
        )

        trainer.train(epochs=args.epochs, save_path=args.model_path)

    if args.mode in ['eval', 'both']:
        print("\nEVALUATION")
        _, _, test_loader = data_pipeline.create_dataloaders()

        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")

        evaluator = EvalPipeline(model=model, device=device)
        evaluator.evaluate(test_loader)


if __name__ == "__main__":
    main()