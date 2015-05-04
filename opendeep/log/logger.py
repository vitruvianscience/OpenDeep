"""
Configuring the logger for our example needs. By default in the logging_config.json file,
this will print logging levels of info and higher to log files in the logs/ directory. Debug goes to console.
"""
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "opendeep-dev@googlegroups.com"

# standard libraries
import os
import logging
import logging.config
import json
# internal references
from opendeep.utils.file_ops import mkdir_p


def get_root_logger():
    """
    Grabs the logger instance for the root of the OpenDeep package.

    Returns
    -------
    logger
        The logger for the root of the OpenDeep package.
    """
    return logging.getLogger(__name__.split('.')[0])


def config_root_logger(config_file='logging_config.json'):
    """
    Configures the root logger (returned from get_root_logger()) to the specifications in the JSON file `config_file`.

    Parameters
    ----------
    config_file : str
        The string path to the configuration JSON file to use.
    """
    # this could be called from scripts anywhere, but we want to keep the log-related items in this directory.
    # therefore, change the cwd to this file's directory and then change back at the end.
    prevdir = os.path.realpath(os.getcwd())
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    # load the basic parameters from the JSON configuration file
    # config_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], config_file)

    path = config_file
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
                                logger.exception('Exception in creating the directory for a logging handler! '
                                                 'Path was {0!s}'.format(os.path.realpath(dirs)))
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
        logger.warning("Could not find configuration file for logger! Was looking for {0!s}. "
                       "Using basicConfig instead...".format(os.path.realpath(path)))

    # change the directory to the calling file's working directory
    os.chdir(prevdir)


def delete_root_logger():
    """
    Deletes the root logger (returned from get_root_logger()). This removes all existing handlers for the logger,
    which effectively renders it useless.
    """
    # get rid of all the existing handlers - effectively renders the logger useless
    root_logger = get_root_logger()
    while root_logger.handlers:
        root_logger.handlers.pop()