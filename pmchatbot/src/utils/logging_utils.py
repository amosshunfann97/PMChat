from config.settings import Config

config = Config()

print("Current LOG_LEVEL:", config.LOG_LEVEL)

def log(message, level="info"):
    levels = ["debug", "info", "warning", "error"]
    current = levels.index(config.LOG_LEVEL) if config.LOG_LEVEL in levels else 1
    msg_level = levels.index(level) if level in levels else 1
    if msg_level >= current:
        print(message)