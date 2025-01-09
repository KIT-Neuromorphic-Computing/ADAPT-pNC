import logging
import os

def GetMessageLogger(args, setup):
    if not os.path.exists(f'{args.logfilepath}/'):
        os.makedirs(f'{args.logfilepath}/')

    LOG_FORMAT = "%(asctime)s - %(levelname)s -\n%(message)s\n"
    formatter = logging.Formatter(LOG_FORMAT)

    log_file = f'{args.logfilepath}/{setup}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    SETUP = vars(args)
    msg = ''
    for k, v in SETUP.items():
        msg = msg + str(k).ljust(20, ' ') + str(v) + '\n'
    
    logger_name = f'msglogger_{setup}'
    msglogger = logging.getLogger(log_file)
    msglogger.addHandler(file_handler)

    if args.loglevel.lower() == 'info':
        msglogger.setLevel(logging.INFO)
    elif args.loglevel.lower() == 'debug':
        msglogger.setLevel(logging.DEBUG)

    msglogger.propagate = False
    msglogger.info(msg)
    return msglogger

def CloseLogger(msglogger):
    for handler in msglogger.handlers:
        msglogger.removeHandler(handler)
        handler.close()
    logging.shutdown()
    
def PrintParser(args):
    args = vars(args)
    for k, v in args.items():
        print(str(k).ljust(20, ' ') + str(v))
        
        
