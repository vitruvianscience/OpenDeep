from opendeep.models.multi_layer.convolutional_network import AlexNet

if __name__ == '__main__':
    # set up the logging environment to display outputs (optional)
    # although this is recommended over print statements everywhere
    import logging
    import opendeep.log.logger as logger
    logger.config_root_logger()
    log = logging.getLogger(__name__)
    log.info("Creating AlexNet!")

    alexnet = AlexNet()

    del alexnet