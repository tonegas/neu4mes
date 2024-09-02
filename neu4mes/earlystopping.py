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
            