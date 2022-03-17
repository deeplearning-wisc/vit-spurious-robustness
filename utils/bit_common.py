import argparse
import logging
import logging.config
import os

import utils.bit_hyperrule as bit_hyperrule


def setup_logger(args):
  """Creates and returns a fancy logger."""
  os.makedirs(os.path.join("logs", args.name, args.dataset, args.model_arch, args.model_type), exist_ok=True)
  logging.config.dictConfig({
      "version": 1,
      "disable_existing_loggers": False,
      "formatters": {
          "standard": {
              "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
          },
      },
      "handlers": {
          "stderr": {
              "level": "INFO",
              "formatter": "standard",
              "class": "logging.StreamHandler",
              "stream": "ext://sys.stderr",
          },
          "logfile": {
              "level": "DEBUG",
              "formatter": "standard",
              "class": "logging.FileHandler",
              "filename": os.path.join("logs", args.name, args.dataset, args.model_arch, args.model_type, "train.log"),
              "mode": "a",
          }
      },
      "loggers": {
          "": {
              "handlers": ["stderr", "logfile"],
              "level": "DEBUG",
              "propagate": True
          },
      }
  })
  logger = logging.getLogger(__name__)
  logger.flush = lambda: [h.flush() for h in logger.handlers]
  logger.info(args)
  return logger
