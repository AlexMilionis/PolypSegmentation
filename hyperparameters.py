from torch.nn import BCEWithLogitsLoss
class Hyperparameters:

    TRAIN_RATIO = 0.8
    EPOCHS = 1
    BATCH_SIZE = 32
    WEIGHT_DECAY = 0.00001
    LEARNING_RATE = 0.01
    # LOSS_FUNCTION = 'binary_crossentropy'
    LOSS_FUNCTIONS = {
        'binary_crossentropy': BCEWithLogitsLoss(),
        # 'dice_loss': DiceLoss(),  # Example for a custom Dice Loss
        # 'combined_loss': CombinedLoss()  # Example for custom combined losses
    }