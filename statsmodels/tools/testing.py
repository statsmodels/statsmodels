import os
import sys

class SilenceTestOutput(object):
    """
    Temporarily redirect system output to file like objects.
    Default is to redirect to `os.devnull`, which just mutes output, `stdout`
    and `stderr`.
    """

    def __init__(self, stdout=None, stderr=None):
        if stdout is None:
            stdout = open(os.devnull, 'w')
        if stderr is None:
            stderr = open(os.devnull, 'w')

        self.__stdout = stdout
        self.__stderr = stderr
        self.__redirected = False

    def __enter__(self):
        self.redirect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unredirect()

    def redirect(self):
        self.old_stdout = sys.stdout
        self.old_stdout.flush()
        self.old_stderr = sys.stderr
        self.old_stderr.flush()
        sys.stdout = self.__stdout
        sys.stderr = self.__stderr
        self.__redirected = True

    def unredirect(self):
        if not self.__redirected:
            return
        try:
            self.__stdout.flush()
            self.__stdout.close()
        except ValueError:
            # already closed?
            pass
        try:
            self.__stderr.flush()
            self.__stderr.close()
        except ValueError:
            # already closed?
            pass

        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

    def flush(self):
        if self.__redirected:
            try:
                self.__stdout.flush()
            except:
                pass
            try:
                self.__stderr.flush()
            except:
                pass

