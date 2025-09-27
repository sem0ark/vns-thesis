import logging

import src.examples.moacbw.vns

import src.examples.mokp.pymoo_builtin
import src.examples.mokp.vns
from src.cli.cli import CLI

logger = logging.getLogger()


def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    cli = CLI()

    src.examples.mokp.pymoo_builtin.register_cli(cli)
    src.examples.mokp.vns.register_cli(cli)

    src.examples.moacbw.vns.register_cli(cli)

    cli.run()
