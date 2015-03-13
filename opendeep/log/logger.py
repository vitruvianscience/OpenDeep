"""
.. module:: logger

Configuring the logger for our example needs. This will print info and up to log files. Debug goes to console.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard libraries
import os
import logging
import logging.config
import json
# internal references
from opendeep.utils.file_ops import mkdir_p


def get_root_logger():
    return logging.getLogger(__name__.split('.')[0])


def config_root_logger():
    # this could be called from scripts anywhere, but we want to keep the log-related items in this directory.
    # therefore, change the cwd to this file's directory and then change back at the end.
    prevdir = os.path.realpath(os.getcwd())
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    # load the basic parameters from the JSON configuration file
    # config_file_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'logging_config.json')
    config_file_path = 'logging_config.json'

    path = config_file_path
    env_key = 'LOG_CFG'
    value = os.getenv(env_key, None)
    if value:
        path = value
    # if the configuration exists
    init = True
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = json.load(f)
            except:
                logging.basicConfig(level=logging.DEBUG)
                logger = get_root_logger()
                logger.exception('Exception in reading the JSON logging config file!')
                logger.warning('Anyway, loading the basicConfig for the logger instead.')
                init = False

        if init:
            # make the file paths to the log files
            for handler in config.get('handlers', None):
                if handler is not None:
                    path = config.get('handlers').get(handler).get('filename')
                    if path is not None:
                        path = os.path.normpath(path)
                        (dirs, _) = os.path.split(path)
                        if len(dirs) is not 0:
                            # dirs = os.path.join(os.path.split(os.path.realpath(__file__))[0], dirs)
                            try:
                                mkdir_p(dirs)
                            except:
                                logging.basicConfig(level=logging.DEBUG)
                                logger = get_root_logger()
                                logger.exception('Exception in creating the directory for a logging handler! Path was {0!s}'.format(os.path.realpath(dirs)))
                                logger.warning('Anyway, loading the basicConfig for the logger instead.')
                                init = False

            # load the configuration into the logging module
            if init:
                try:
                    logging.config.dictConfig(config)
                except:
                    logging.basicConfig(level=logging.DEBUG)
                    logger = get_root_logger()
                    logger.exception('Exception in loading the JSON logging config file to the logging module!')
                    logger.warning('Anyway, loading the basicConfig for the logger instead.')

    # otherwise, couldn't find the configuration file
    else:
        logging.basicConfig(level=logging.DEBUG)
        logger = get_root_logger()
        logger.warning("Could not find configuration file for logger! Was looking for {0!s}. Using basicConfig instead...".format(os.path.realpath(path)))

    # change the directory to the calling file's working directory
    os.chdir(prevdir)


def delete_root_logger():
    # get rid of all the existing handlers - effectively renders the logger useless
    root_logger = get_root_logger()
    while root_logger.handlers:
        root_logger.handlers.pop()