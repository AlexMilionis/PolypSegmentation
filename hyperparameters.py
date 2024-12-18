from torch.nn import BCEWithLogitsLoss

class Hyperparameters:

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    EPOCHS = 3
    BATCH_SIZE = 32
    WEIGHT_DECAY = 0.00001
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2

    # LOSS_FUNCTION = 'binary_crossentropy'
    LOSS_FUNCTIONS = {
        'binary_crossentropy_with_logits': BCEWithLogitsLoss(),
        # 'dice_loss': DiceLoss(),  # Example for a custom Dice Loss
        # 'combined_loss': CombinedLoss()  # Example for custom combined losses
    }