
import os
import yaml
import datetime

def checkOutputDirectoryAndCreate(output_folder):
    if not os.path.exists('result/' + output_folder):
        os.makedirs('result/' + output_folder)

def loadConfig(path):
    f = open(path)
    config = yaml.load(f, Loader=yaml.FullLoader)
    tz = datetime.timezone(datetime.timedelta(hours=+8))
    config["start_time"] = datetime.datetime.now(tz=tz).strftime("%m-%d_%H:%M")
    return config

def showConfig(config):
    print(yaml.dump(config, indent=4, default_flow_style=False))

def dumpConfig(config):
    with open("./result/" + config["output_folder"] + "/config.yaml", 'w') as outputFile:
        yaml.dump(config, outputFile)
    return