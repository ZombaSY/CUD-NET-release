
def regularization_loss(model, p=2, loss_lambda=1e-5):
    loss = 0

    for w in model.parameters():
        loss += w.norm(p)

    return loss * loss_lambda

