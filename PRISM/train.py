import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .loss import WeightedBinaryClassificationLoss
from .utils import calculate_metrics, free_memory, update_importance_csv


def train_and_validate(
    model,
    optimizer,
    data_manager,
    config,
    save_results=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    patience=10,
    min_delta=0.001,
    use_l1_reg=True,
    model_save=False,
):
    model.to(device)
    loss_fn = WeightedBinaryClassificationLoss(
        config["IMPORTANCE_DECAY"],
        config["IMPORTANCE_REGULARIZATION_WEIGHT"],
        use_l1_reg=use_l1_reg,
    )
    train_metrics, validation_metrics = [], []

    best_val_loss = float("inf")
    counter = 0
    best_model = None

    use_amp = device.startswith("cuda")
    scaler = GradScaler() if use_amp else None

    for epoch in range(config["NUM_EPOCHS"]):
        train_results = run_epoch(
            model,
            optimizer,
            data_manager.train_gene_loader,
            data_manager.train_target_loader,
            loss_fn,
            device,
            is_training=True,
            save_results=save_results,
            scaler=scaler,
            use_amp=use_amp,
        )
        train_metrics.append(train_results)

        with torch.no_grad():
            val_results = run_epoch(
                model,
                optimizer,
                data_manager.valid_gene_loader,
                data_manager.valid_target_loader,
                loss_fn,
                device,
                is_training=False,
                save_results=save_results,
                use_amp=use_amp,
            )
        validation_metrics.append(val_results)

        print(
            f"Epoch [{epoch+1}/{config['NUM_EPOCHS']}], "
            f"Train Loss: {train_results['loss']:.4f}, "
            f"Train Accuracy: {train_results['accuracy']:.4f}, "
            f"Validation Loss: {val_results['loss']:.4f}, "
            f"Validation Accuracy: {val_results['accuracy']:.4f}"
        )

        # Early stopping logic and best model saving
        if model_save:
            model_name = config["MODEL_NAME"]
            best_model_path = f"best_model_{model_name}.pth"

            if val_results["loss"] < best_val_loss - min_delta:
                best_val_loss = val_results["loss"]
                counter = 0
                best_model = model.state_dict()
                torch.save(best_model, best_model_path)
                print(f"New best model saved at epoch {epoch+1}")
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(torch.load(best_model_path))
                break

    return train_metrics, validation_metrics, model


def run_epoch(
    model,
    optimizer,
    gene_loader,
    target_loader,
    loss_fn,
    device,
    is_training,
    save_results,
    scaler=None,
    use_amp=False,
):
    model.train() if is_training else model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []

    for batch, target in tqdm(
        zip(gene_loader, target_loader),
        desc="Training" if is_training else "Validating",
        total=len(gene_loader),
    ):
        if use_amp:
            with autocast(device_type="cuda", enabled=use_amp):
                loss, predictions = process_batch(
                    model,
                    batch,
                    target,
                    loss_fn,
                    device,
                    "Train" if is_training else "Val",
                    save_results,
                )
        else:
            loss, predictions = process_batch(
                model,
                batch,
                target,
                loss_fn,
                device,
                "Train" if is_training else "Val",
                save_results,
            )

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        all_targets.append(target.to("cpu"))
        all_predictions.append(predictions.to("cpu"))

    all_targets = torch.cat(all_targets).cpu().detach().numpy()
    all_predictions = torch.cat(all_predictions).cpu().detach().numpy()

    avg_loss = total_loss / len(gene_loader)
    metrics = calculate_metrics(all_targets, all_predictions)
    metrics["loss"] = avg_loss
    return metrics


def process_batch(model, batch, target, loss_fn, device, data_type, save_results):
    x = batch.x.to(device, non_blocking=True)
    edge_index = batch.edge_index.to(device, non_blocking=True)
    edge_attr = batch.edge_attr.to(device, non_blocking=True)
    initial_importance = batch.dti.to(device, non_blocking=True)
    batch_index = batch.batch.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    logits, pred_importance = model(
        x, edge_index, edge_attr, initial_importance, batch_index
    )

    if save_results:
        update_importance_csv(pred_importance.detach().cpu(), batch, data_type)

    loss = loss_fn(target, logits.squeeze(), pred_importance, initial_importance)
    return loss, torch.sigmoid(logits.squeeze())


def predict(
    model,
    data_manager,
    save_results=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    predictions, targets, importances = collect_predictions(
        model, data_manager, save_results, device
    )
    metrics = calculate_metrics(targets, predictions)
    return metrics, predictions, targets, importances


def collect_predictions(model, data_manager, save_results, device):
    predictions, targets, importances = [], [], []
    with torch.no_grad():
        for batch, target in tqdm(
            zip(data_manager.test_gene_loader, data_manager.test_target_loader),
            desc="Predicting",
            total=len(data_manager.test_gene_loader),
        ):
            logits, pred_importance = process_prediction_batch(model, batch, device)
            pred = torch.sigmoid(logits)  # シグモイド関数を適用
            target = target.to(device, non_blocking=True)
            predictions.append(pred.squeeze().cpu())
            targets.append(target.cpu())
            importances.append(pred_importance.detach().cpu())
            if save_results:
                update_importance_csv(pred_importance.detach().cpu(), batch, "Test")
    return (
        torch.cat(predictions).cpu().detach().numpy(),
        torch.cat(targets).cpu().detach().numpy(),
        torch.cat(importances),
    )


def process_prediction_batch(model, batch, device):
    x = batch.x.to(device, non_blocking=True)
    edge_index = batch.edge_index.to(device, non_blocking=True)
    edge_attr = batch.edge_attr.to(device, non_blocking=True)
    initial_importance = batch.dti.to(device, non_blocking=True)
    batch_index = batch.batch.to(device, non_blocking=True)
    return model(x, edge_index, edge_attr, initial_importance, batch_index)


def evaluate(
    model,
    data_loader,
    loss_fn,
    save_results=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    with torch.no_grad():
        result = run_epoch(
            model,
            None,
            data_loader,
            None,
            loss_fn,
            device,
            is_training=False,
            save_results=save_results,
        )
    return result


# ------------------------------------------------------------


def setup_training(model, config, device, use_l1_reg=True):
    model.to(device)
    loss_fn = WeightedBinaryClassificationLoss(
        config["IMPORTANCE_DECAY"],
        config["IMPORTANCE_REGULARIZATION_WEIGHT"],
        use_l1_reg=use_l1_reg,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    use_amp = device.startswith("cuda")
    scaler = GradScaler() if use_amp else None
    return model, loss_fn, optimizer, use_amp, scaler


def train_epoch(
    model, optimizer, data_manager, loss_fn, device, save_results, scaler, use_amp
):
    return run_epoch(
        model,
        optimizer,
        data_manager.train_gene_loader,
        data_manager.train_target_loader,
        loss_fn,
        device,
        is_training=True,
        save_results=save_results,
        scaler=scaler,
        use_amp=use_amp,
    )


def validate_epoch(model, data_manager, loss_fn, device, save_results, use_amp):
    with torch.no_grad():
        return run_epoch(
            model,
            None,
            data_manager.valid_gene_loader,
            data_manager.valid_target_loader,
            loss_fn,
            device,
            is_training=False,
            save_results=save_results,
            use_amp=use_amp,
        )


def update_best_model(model, val_loss, best_val_loss, min_delta, counter, best_model):
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
        best_model = model.state_dict()
    else:
        counter += 1
    return best_val_loss, counter, best_model


def check_early_stopping(counter, patience, epoch, model, best_model):
    if counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        model.load_state_dict(best_model)
        return True
    return False


def print_epoch_results(
    epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, test_acc
):
    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_acc:.4f}, "
        f"Validation Loss: {val_loss:.4f}, "
        f"Validation Accuracy: {val_acc:.4f}, "
        f"Test Accuracy: {test_acc:.4f}"
    )


def run_full_cycle(
    model,
    data_manager,
    config,
    save_results=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    patience=5,
    min_delta=0.001,
    use_l1_reg=True,
):
    model, loss_fn, optimizer, use_amp, scaler = setup_training(
        model, config, device, use_l1_reg
    )
    train_metrics, validation_metrics, prediction_metrics = [], [], []

    best_val_loss = float("inf")
    counter = 0
    best_model = None

    for epoch in range(config["NUM_EPOCHS"]):
        # Training
        train_results = train_epoch(
            model,
            optimizer,
            data_manager,
            loss_fn,
            device,
            save_results,
            scaler,
            use_amp,
        )
        train_metrics.append(train_results)
        train_loss, train_acc = train_results["loss"], train_results["accuracy"]
        del train_results
        free_memory()

        # Validation
        val_results = validate_epoch(
            model, data_manager, loss_fn, device, save_results, use_amp
        )
        validation_metrics.append(val_results)
        val_loss, val_acc = val_results["loss"], val_results["accuracy"]
        del val_results
        free_memory()

        # Prediction
        pred_metrics, predictions, targets, importances = predict(
            model,
            data_manager,
            save_results=save_results,
            device=device,
        )
        prediction_metrics.append(pred_metrics)
        test_acc = pred_metrics["accuracy"]
        del pred_metrics, predictions, targets, importances
        free_memory()

        print_epoch_results(
            epoch,
            config["NUM_EPOCHS"],
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            test_acc,
        )

        # Early stopping logic
        best_val_loss, counter, best_model = update_best_model(
            model, val_loss, best_val_loss, min_delta, counter, best_model
        )
        if check_early_stopping(counter, patience, epoch, model, best_model):
            break

    return train_metrics, validation_metrics, prediction_metrics, model
