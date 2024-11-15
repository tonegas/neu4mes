# Early stopping functions:
# The functions return True if the training should stop

# "Classical" early stopping based on the validation loss:
# Stop if the validation loss has not improved for a certain number of epochs
def early_stop_patience(train_losses, val_losses, params):
  patience = params['patience'] if 'patience' in params.keys() else 50
  if val_losses:
    losses = val_losses
  else:
    # if there is no validation set, use the training losses
    losses = train_losses

  if 'error' in params.keys():
    # if the type of loss to be used is provided by the user
    losses_use = losses[params['error']]
  else:
    # take the mean of all the losses for all the keys of the dictionary
    import numpy as np
    losses_use = [np.mean([losses[key][index] for key in losses.keys()]) for index in range(len(losses[list(losses.keys())[0]]))]
  if len(losses_use) > patience:
    # index of the minimum validation loss
    min_val_loss_index = losses_use.index(min(losses_use))
    # check if the patience has been exceeded
    if min_val_loss_index < len(losses_use) - patience:
      return True
  return False


def select_best_model(train_losses, val_losses, params):
    if val_losses:
        losses = val_losses
    else:
        # if there is no validation set, use the training losses
        losses = train_losses
    import numpy as np
    losses_use = [np.mean([losses[key][index] for key in losses.keys()]) for index in
                  range(len(losses[list(losses.keys())[0]]))]
    if len(losses_use)-1 == losses_use.index(min(losses_use)):
        return True
    else:
        return False


def mean_stopping(train_losses, val_losses, params):
    tol = params['tol'] if 'tol' in params.keys() else 0.001
    if val_losses:
        for (train_loss_name, train_loss_value), (val_loss_name, val_loss_value) in zip(train_losses.items(), val_losses.items()):
            if abs(train_loss_value[-1] - val_loss_value[-1]) < tol:
                return True
    else:
        for loss_name, loss_value in train_losses.items():
            if loss_value[-1] < tol:
                return True
    return False

def standard_early_stopping(train_losses, val_losses, params):
    n = params['tol'] if 'tol' in params.keys() else 10
    if val_losses:
        for (_, train_loss_value), (_, val_loss_value) in zip(train_losses.items(), val_losses.items()):
            if (len(train_loss_value) <= n) and (len(val_loss_value) <= n):
                return False
            
            tol = 0.0
            for train_loss, val_loss in zip(train_loss_value[-n:], val_loss_value[-n:]):
                if abs(train_loss - val_loss) > tol:
                    tol = abs(train_loss - val_loss)
                else:
                    return False
    else:
        for _, loss_value in train_losses.items():
            if (len(loss_value) <= n):
                return False
            
            tol = loss_value[-n]
            for loss in loss_value[-n+1:]:
                if loss < tol:
                    return False
                
    return True
            