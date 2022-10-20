try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


def logFeature(feature):
    ## wandb
    if (wandb != None):
        logMsg = {}
        logMsg["Feature/max"] = feature.max()
        logMsg["Feature/min"] = feature.min()
        logMsg["Feature/mean"] = feature.mean()
        logMsg["Feature/var"] = feature.var()
        logMsg["Feature/var_dim"] = feature.var(dim=0).mean()
        wandb.log(logMsg)
    return