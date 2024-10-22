# Copyright (c) OpenMMLab. All rights reserved.
import logging
import sys
from logging import Logger, LogRecord
from typing import Optional, Union

from termcolor import colored


class FilterDuplicateWarning(logging.Filter):
    """Filter the repeated warning message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name: str = 'fusion'):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """Filter the repeated warning message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class ColorfulFormatter(logging.Formatter):
    _color_mapping: dict = dict(
        ERROR='red', WARNING='yellow', INFO='white', DEBUG='green'
    )

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (
            not color and blink
        ), 'blink should only be available when color is True'
        # Get prefix format according to color.
        error_prefix = self._get_prefix('ERROR', color, blink=True)
        warn_prefix = self._get_prefix('WARNING', color, blink=True)
        info_prefix = self._get_prefix('INFO', color, blink)
        debug_prefix = self._get_prefix('DEBUG', color, blink)

        # Config output format.
        self.err_format = f'%(asctime)s - %(name)s - {error_prefix} - %(pathname)s - %(funcName)s - %(lineno)d - %(message)s'
        self.warn_format = f'%(asctime)s - %(name)s - {warn_prefix} - %(message)s'
        self.info_format = f'%(asctime)s - %(name)s - {info_prefix} - %(message)s'
        self.debug_format = f'%(asctime)s - %(name)s - {debug_prefix} - %(message)s'

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        """Get the prefix of the target log level.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.
            blink (bool): Whether the prefix will blink.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            attrs = ['underline']
            if blink:
                attrs.append('blink')
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """Override the `logging.Formatter.format`` method `. Output the
        message according to the specified log level.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class FusionLogger(Logger):
    """Formatted logger used to record messages.

    ``MMLogger`` can create formatted logger to log message with different
    log levels and get instance in the same way as ``ManagerMixin``.
    ``MMLogger`` has the following features:

    - Distributed log storage, ``MMLogger`` can choose whether to save log of
      different ranks according to `log_file`.
    - Message with different log levels will have different colors and format
      when displayed on terminal.

    Note:
        - The `name` of logger and the ``instance_name`` of ``MMLogger`` could
          be different. We can only get ``MMLogger`` instance by
          ``MMLogger.get_instance`` but not ``logging.getLogger``. This feature
          ensures ``MMLogger`` will not be incluenced by third-party logging
          config.
        - Different from ``logging.Logger``, ``MMLogger`` will not log warning
          or error message without ``Handler``.

    Examples:
        >>> logger = MMLogger.get_instance(name='MMLogger',
        >>>                                logger_name='Logger')
        >>> # Although logger has name attribute just like `logging.Logger`
        >>> # We cannot get logger instance by `logging.getLogger`.
        >>> assert logger.name == 'Logger'
        >>> assert logger.instance_name = 'MMLogger'
        >>> assert id(logger) != id(logging.getLogger('Logger'))
        >>> # Get logger that do not store logs.
        >>> logger1 = MMLogger.get_instance('logger1')
        >>> # Get logger only save rank0 logs.
        >>> logger2 = MMLogger.get_instance('logger2', log_file='out.log')
        >>> # Get logger only save multiple ranks logs.
        >>> logger3 = MMLogger.get_instance('logger3', log_file='out.log',
        >>>                                 distributed=True)

    Args:
        name (str): Global instance name.
        logger_name (str): ``name`` attribute of ``Logging.Logger`` instance.
            If `logger_name` is not defined, defaults to 'mmengine'.
        log_file (str, optional): The log filename. If specified, a
            ``FileHandler`` will be added to the logger. Defaults to None.
        log_level (str): The log level of the handler. Defaults to
            'INFO'. If log level is 'DEBUG', distributed logs will be saved
            during distributed training.
        file_mode (str): The file mode used to open log file. Defaults to 'w'.
        distributed (bool): Whether to save distributed logs, Defaults to
            false.
    """

    def __init__(
        self,
        logger_name='fusion',
        log_file: Optional[str] = None,
        log_level: Union[int, str] = 'INFO',
        file_mode: str = 'w',
    ):
        super(FusionLogger, self).__init__(logger_name)
        # Get rank in DDP mode.
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]

        # Config stream_handler. If `rank != 0`. stream_handler can only export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second timestamp.
        stream_handler.setFormatter(
            ColorfulFormatter(color=True, datefmt='%m/%d %H:%M:%S')
        )
        # Only rank0 `StreamHandler` will log messages below error level.
        stream_handler.setLevel(log_level)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        self.handlers.append(stream_handler)

        if log_file is not None:
            # Save multi-ranks logs if distributed is True. The logs of rank0
            # will always be saved.
            # Here, the default behaviour of the official logger is 'a'.
            # Thus, we provide an interface to change the file mode to
            # the default behaviour. `FileHandler` is not supported to
            # have colors, otherwise it will appear garbled.
            file_handler = logging.FileHandler(log_file, file_mode)
            # `StreamHandler` record year, month, day hour, minute,
            # and second timestamp. file_handler will only record logs
            # without color to avoid garbled code saved in files.
            file_handler.setFormatter(
                ColorfulFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S')
            )
            file_handler.setLevel(log_level)
            file_handler.addFilter(FilterDuplicateWarning(logger_name))
            self.handlers.append(file_handler)
        self._log_file = log_file

    @property
    def log_file(self):
        return self._log_file

    # def callHandlers(self, record: LogRecord) -> None:
    #     """Pass a record to all relevant handlers.

    #     Override ``callHandlers`` method in ``logging.Logger`` to avoid
    #     multiple warning messages in DDP mode. Loop through all handlers of
    #     the logger instance and its parents in the logger hierarchy. If no
    #     handler was found, the record will not be output.

    #     Args:
    #         record (LogRecord): A ``LogRecord`` instance contains logged
    #             message.
    #     """
    #     for handler in self.handlers:
    #         if record.levelno >= handler.level:
    #             handler.handle(record)

    def setLevel(self, level):
        """Set the logging level of this logger.

        If ``logging.Logger.selLevel`` is called, all ``logging.Logger``
        instances managed by ``logging.Manager`` will clear the cache. Since
        ``MMLogger`` is not managed by ``logging.Manager`` anymore,
        ``MMLogger`` should override this method to clear caches of all
        ``MMLogger`` instance which is managed by :obj:`ManagerMixin`.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)  # type: ignore


def get_logger(
    logger_name='fusion',
    log_file: Optional[str] = None,
    log_level: Union[int, str] = 'INFO',
    file_mode: str = 'w',
):
    logger = logging.getLogger(logger_name)
    if isinstance(log_level, str):
        log_level = logging._nameToLevel[log_level]

    # Config stream_handler. If `rank != 0`. stream_handler can only export ERROR logs.
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    # `StreamHandler` record month, day, hour, minute, and second timestamp.
    stream_handler.setFormatter(ColorfulFormatter(color=True, datefmt='%m/%d %H:%M:%S'))
    # Only rank0 `StreamHandler` will log messages below error level.
    stream_handler.setLevel(log_level)
    stream_handler.addFilter(FilterDuplicateWarning(logger_name))
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(
            ColorfulFormatter(color=False, datefmt='%Y/%m/%d %H:%M:%S')
        )
        file_handler.setLevel(log_level)
        file_handler.addFilter(FilterDuplicateWarning(logger_name))
        logger.addHandler(file_handler)
    return logger


if __name__ == '__main__':
    logger = get_logger(log_file='log/test.log', log_level='DEBUG', file_mode='w')
    logger.info('test_info')
    logger.debug('test_debug')
    logger.warning('test_warning')
    logger.error('test_error')
    logger.critical('test_critical')
