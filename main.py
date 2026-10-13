import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from dataset import resolve_data_dir, get_loaders
from model import PneumoniaDenseNet
from gradcam import save_gradcam

OUTPUT_DIR = Path('outputs')
CHECKPOINT = OUTPUT_DIR / 'best_model.pth'

def train(loaders, sets, device, epochs):
    labels = [label for _, label in sets['train'].samples]
    negatives, positives = labels.count(0), labels.count(1)
    weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    model = PneumoniaDenseNet(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        for phase in ('train', 'val'):
            model.train(phase == 'train')
            loss_sum = correct = count = 0
            for images, targets in loaders[phase]:
                images = images.to(device)
                targets = targets.to(device).float().unsqueeze(1)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(images)
                    loss = criterion(logits, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                loss_sum += loss.item() * images.size(0)
                correct += ((torch.sigmoid(logits) >= 0.5) == targets.bool()).sum().item()
                count += images.size(0)
            avg_loss = loss_sum / count
            print(f'{phase}: loss={avg_loss:.4f}, accuracy={correct / count:.4f}')
            if phase == 'val' and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), CHECKPOINT)
                print(f'Saved {CHECKPOINT}')

def evaluate(model, loader, device):
    model.eval()
    truth, probabilities = [], []
    with torch.no_grad():
        for images, labels in loader:
            scores = torch.sigmoid(model(images.to(device))).cpu().flatten().tolist()
            probabilities.extend(scores)
            truth.extend(labels.tolist())
    predictions = [int(p >= 0.5) for p in probabilities]
    print(classification_report(truth, predictions, target_names=['NORMAL', 'PNEUMONIA'], zero_division=0))
    if len(set(truth)) == 2:
        print(f'ROC-AUC: {roc_auc_score(truth, probabilities):.4f}')
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'], ax=ax)
    ax.set(xlabel='Predicted', ylabel='Actual', title='Test confusion matrix')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    print('Confusion matrix saved.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None, help='Folder containing train, val and test; downloads if omitted')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--image', default=None, help='X-ray image for Grad-CAM')
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_dir = resolve_data_dir(args.data_dir)
    print('Dataset:', data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    loaders, sets = get_loaders(data_dir, args.batch_size)
    if not args.eval_only:
        train(loaders, sets, device, args.epochs)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'{CHECKPOINT} not found; run training first.')
    model = PneumoniaDenseNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    evaluate(model, loaders['test'], device)
    sample = Path(args.image) if args.image else next((data_dir / 'test' / 'PNEUMONIA').iterdir())
    if not sample.is_file():
        raise FileNotFoundError(sample)
    save_gradcam(model, sample, device, OUTPUT_DIR / 'gradcam.png')

if __name__ == '__main__':
    main()
