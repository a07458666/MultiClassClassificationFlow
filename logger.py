try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def set_wandb(args):
    ## wandb
    if (wandb != None):
        wandb.init(project="Multi-Label", entity="andy-su", name=args.exp_name)
        wandb.config.update(args)
        wandb.define_metric("test/mAP", summary="max")
        wandb.define_metric("test/mAP_flow", summary="max")
        wandb.define_metric("val/mAP", summary="max")
        wandb.define_metric("val/mAP_flow", summary="max")
        wandb.define_metric("loss", summary="min")
        wandb.run.log_code(".")
    return

def log_mAP(phase, epoch, mAP, mAP_flow = None, mAP_ema = None, mAP_flow_ema = None):
    if (wandb != None):
        logMsg = {}
        logMsg["epoch"] = epoch
        logMsg[f"{phase}/mAP"] = mAP
        if mAP_flow != None:
            logMsg[f"{phase}/mAP_flow"] = mAP_flow
        if mAP_ema != None:
            logMsg[f"{phase}/mAP_ema"] = mAP_ema
        if mAP_flow_ema != None:
            logMsg[f"{phase}/mAP_flow_ema"] = mAP_flow_ema
        wandb.log(logMsg)
    return 