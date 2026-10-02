from jevlite.log import logger, setup_logging


def test_setup_logging_writes_console_and_file(tmp_path, capsys):
    path = tmp_path / "run" / "run.log"
    setup_logging(path)
    logger.info("step {} loss {:.2f}", 3, 0.5)
    logger.debug("only in the file")
    setup_logging(path)  # re-running a notebook cell must not duplicate sinks
    logger.info("again")
    out = capsys.readouterr().out
    assert "step 3 loss 0.50" in out and "only in the file" not in out and out.count("again") == 1
    logger.remove()
    text = path.read_text()
    assert "step 3 loss 0.50" in text and "only in the file" in text
