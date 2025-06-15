import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class EvalPipeline:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.classes = ('dog', 'horse', 'elephant', 'butterfly', 'chicken',
                        'cat', 'cow', 'sheep', 'spider', 'squirrel')

    def evaluate(self, test_loader):
        self.model.eval()

        all_predictions = []
        all_targets = []
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = torch.nn.CrossEntropyLoss()

        print("Evaluating model")
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Testing"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Correct predictions: {correct}/{total}")

        self.generate_classification_report(all_targets, all_predictions)
        self.plot_confusion_matrix(all_targets, all_predictions)
        self.show_sample_predictions(test_loader)

        return test_acc, test_loss

    def generate_classification_report(self, y_true, y_pred):
        report = classification_report(
            y_true, y_pred,
            target_names=self.classes,
            digits=4
        )

        print("\nDetailed Classification Report:")
        print("=" * 60)
        print(report)

        with open('results/classification_report.txt', 'w') as f:
            f.write("Animals-10 Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(report)

        print("Classification report saved to results/classification_report.txt")

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix - Animals-10 Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Confusion matrix saved to results/confusion_matrix.png")

    def show_sample_predictions(self, test_loader, num_samples=8):
        self.model.eval()

        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()

        for i in range(min(num_samples, len(images))):
            img = images[i].cpu()
            img = img * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            img = img + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)

            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_title(f'True: {self.classes[labels[i]]}\n'
                              f'Pred: {self.classes[predicted[i]]}',
                              color='green' if labels[i] == predicted[i] else 'red')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('results/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Sample predictions saved to results/sample_predictions.png")