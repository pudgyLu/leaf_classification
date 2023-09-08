import torch
from tqdm import tqdm


def train_ft(model, train_loader, val_loader, optimizer, criterion,
             n_epochs, model_path, device):
    best_acc = 0.0
    train_step = 0

    for epoch in range(n_epochs):
        # ---------- Training ----------
        model.train()
        train_loss, train_accs = [], []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)
            if train_step % 30 == 0:
                print(f"[ Train | {train_step + 1:03d}] loss = {loss.item():.5f}, acc = {acc:.5f}")
            train_step += 1

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(val_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))
