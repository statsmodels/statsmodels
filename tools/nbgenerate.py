#! /usr/bin/env python
"""
Script to generate notebooks with output from notebooks that don't have
output.
"""

# prefer HTML over rST for now until nbconvert changes drop
OUTPUT = "html"

import os
import io
import sys
import time
import shutil

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "examples",
                                          "notebooks"))

from Queue import Empty

try: # IPython has been refactored
    from IPython.kernel import KernelManager
except ImportError:
    from IPython.zmq.blockingkernelmanager import (BlockingKernelManager as
                                                   KernelManager)

from IPython.nbformat.current import reads, write, NotebookNode

cur_dir = os.path.abspath(os.path.dirname(__file__))

# for conversion of .ipynb -> html/rst

from IPython.config import Config
try:
    from IPython.nbconvert.exporters import HTMLExporter
except ImportError:
    from warnings import warn
    from statsmodels.tools.sm_exceptions import ModuleUnavailableWarning
    warn("Notebook examples not built. You need IPython 1.0.",
         ModuleUnavailableWarning)
    sys.exit(0)

import hash_funcs

class NotebookRunner:
    """
    Paramters
    ---------
    notebook_dir : str
        Path to the notebooks to convert
    extra_args : list
        These are command line arguments passed to start the notebook kernel
    profile : str
        The profile name to use
    timeout : int
        How many seconds to wait for each sell to complete running
    """
    def __init__(self, notebook_dir, extra_args=None, profile=None,
                 timeout=90):
        self.notebook_dir = os.path.abspath(notebook_dir)
        self.profile = profile
        self.timeout = timeout
        km = KernelManager()
        if extra_args is None:
            extra_args = []
        if profile is not None:
            extra_args += ["--profile=%s" % profile]
        km.start_kernel(stderr=open(os.devnull, 'w'),
                        extra_arguments=extra_args)
        try:
            kc = km.client()
            kc.start_channels()
            iopub = kc.iopub_channel
        except AttributeError: # still on 0.13
            kc = km
            kc.start_channels()
            iopub = kc.sub_channel
        shell = kc.shell_channel
        # make sure it's working
        shell.execute("pass")
        shell.get_msg()

        # all of these should be run pylab inline
        shell.execute("%pylab inline")
        shell.get_msg()

        self.kc = kc
        self.km = km
        self.iopub = iopub

    def __iter__(self):
        notebooks = [os.path.join(self.notebook_dir, i)
                     for i in os.listdir(self.notebook_dir)
                     if i.endswith('.ipynb') and 'generated' not in i]
        for ipynb in notebooks:
            with open(ipynb, 'r') as f:
                nb = reads(f.read(), 'json')
            yield ipynb, nb

    def __call__(self, nb):
        return self.run_notebook(nb)

    def run_cell(self, shell, iopub, cell, exec_count):
        outs = []
        shell.execute(cell.input)
        # hard-coded timeout, problem?
        shell.get_msg(timeout=90)
        cell.prompt_number = exec_count # msg["content"]["execution_count"]

        while True:
            try:
                # whats the assumption on timeout here?
                # is it asynchronous?
                msg = iopub.get_msg(timeout=.2)
            except Empty:
                break
            msg_type = msg["msg_type"]
            if msg_type in ["status" , "pyin"]:
                continue
            elif msg_type == "clear_output":
                outs = []
                continue

            content = msg["content"]
            out = NotebookNode(output_type=msg_type)

            if msg_type == "stream":
                out.stream = content["name"]
                out.text = content["data"]
            elif msg_type in ["display_data", "pyout"]:
                for mime, data in content["data"].iteritems():
                    attr = mime.split("/")[-1].lower()
                    # this gets most right, but fix svg+html, plain
                    attr = attr.replace('+xml', '').replace('plain', 'text')
                    setattr(out, attr, data)
                if msg_type == "pyout":
                    out.prompt_number = exec_count #content["execution_count"]
            elif msg_type == "pyerr":
                out.ename = content["ename"]
                out.evalue = content["evalue"]
                out.traceback = content["traceback"]
            else:
                print "unhandled iopub msg:", msg_type

            outs.append(out)

        return outs

    def run_notebook(self, nb):
        """
        """
        shell = self.kc.shell_channel
        iopub = self.iopub
        cells = 0
        errors = 0
        cell_errors = 0
        exec_count = 1

        #TODO: What are the worksheets? -ss
        for ws in nb.worksheets:
            for cell in ws.cells:
                if cell.cell_type != 'code':
                    # there won't be any output
                    continue
                cells += 1
                try:
                    # attaches the output to cell inplace
                    outs = self.run_cell(shell, iopub, cell, exec_count)
                    if outs and outs[-1]['output_type'] == 'pyerr':
                        cell_errors += 1
                    exec_count += 1
                except Exception as e:
                    print "failed to run cell:", repr(e)
                    print cell.input
                    errors += 1
                    continue
                cell.outputs = outs

        print "ran notebook %s" % nb.metadata.name
        print "    ran %3i cells" % cells
        if errors:
            print "    %3i cells raised exceptions" % errors
        else:
            print "    there were no errors in run_cell"
        if cell_errors:
            print "    %3i cells have exceptions in their output" % cell_errors
        else:
            print "    all code executed in the notebook as expected"

    def __del__(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()
        del self.km

def _get_parser():
    try:
        import argparse
    except ImportError:
        raise ImportError("This script only runs on Python >= 2.7")
    parser = argparse.ArgumentParser(description="Convert .ipynb notebook "
                                      "inputs to HTML page with output")
    parser.add_argument("path", type=str, default=SOURCE_DIR, nargs="?",
                        help="path to folder containing notebooks")
    parser.add_argument("--profile", type=str,
                        help="profile name to use")
    parser.add_argument("--timeout", default=90, type=int,
                        metavar="N",
                        help="how long to wait for cells to run in seconds")
    return parser

def nb2html(nb):
    """
    Cribbed from nbviewer
    """
    config = Config()
    config.HTMLExporter.template_file = 'basic'
    config.NbconvertApp.fileext = "html"
    config.CSSHtmlHeaderTransformer.enabled = False

    C = HTMLExporter(config=config)
    return C.from_notebook_node(nb)[0]

def nb2rst(nb, files_dir):
    """
    nb should be a NotebookNode
    """
    #NOTE: This does not currently work. Needs to be update to IPython 1.0.
    config = Config()
    C = ConverterRST(config=config)
    # bastardize how this is supposed to be called
    # either the API is broken, or I'm not using it right
    # why can't I set this using the config?
    C.files_dir = files_dir + "_files"
    if not os.path.exists(C.files_dir):
        os.makedirs(C.files_dir)
    # already parsed into a NotebookNode
    C.nb = nb
    return C.convert()

if __name__ == '__main__':
    rst_target_dir = os.path.join(cur_dir, '..',
                        'docs/source/examples/notebooks/generated/')
    if not os.path.exists(rst_target_dir):
        os.makedirs(rst_target_dir)

    parser = _get_parser()
    arg_ns, other_args = parser.parse_known_args()

    os.chdir(arg_ns.path) # so we execute in notebook dir
    notebook_runner = NotebookRunner(arg_ns.path, other_args, arg_ns.profile,
                                     arg_ns.timeout)
    try:
        for fname, nb in notebook_runner:
            base, ext = os.path.splitext(fname)
            fname_only = os.path.basename(base)
            # check if we need to write
            towrite, filehash = hash_funcs.check_hash(open(fname, "r").read(),
                                                      fname_only)
            if not towrite:
                print "Hash has not changed for file %s" % fname_only
                continue
            print "Writing ", fname_only

            # This edits the notebook cells inplace
            notebook_runner(nb)
            # for debugging writes ipynb file with output
            #new_ipynb = "%s_generated%s" % (base, ".ipynb")
            #with io.open(new_ipynb, "w", encoding="utf-8") as f:
            #    write(nb, f, "json")

            # use nbconvert to convert to rst
            support_file_dir = os.path.join(rst_target_dir,
                                            fname_only+"_files")
            if OUTPUT == "rst":
                new_rst = os.path.join(rst_target_dir, fname_only+".rst")
                rst_out = nb2rst(nb, fname_only)
                # write them to source directory
                if not os.path.exists(rst_target_dir):
                    os.makedirs(rst_target_dir)
                with io.open(new_rst, "w", encoding="utf-8") as f:
                    f.write(rst_out)

                # move support files
                if os.path.exists(fname_only+"_files"):
                    shutil.move(fname_only+"_files",
                            os.path.join(rst_target_dir, fname_only+"_files"))
            elif OUTPUT == "html":
                from notebook_output_template import notebook_template
                new_html = os.path.join(rst_target_dir, fname_only+".rst")
                # get the title out of the notebook because sphinx needs it
                title_cell = nb['worksheets'][0]['cells'].pop(0)
                if title_cell['cell_type'] == 'heading':
                    pass
                elif (title_cell['cell_type'] == 'markdown'
                      and title_cell['source'].strip().startswith('#')):
                    # IPython 3.x got rid of header cells
                    pass
                else:
                    print "Title not in first cell for ", fname_only
                    print "Not generating rST"
                    continue

                html_out = nb2html(nb)
                # indent for insertion into raw html block in rST
                html_out = "\n".join(["   "+i for i in html_out.split("\n")])
                with io.open(new_html, "w", encoding="utf-8") as f:
                    f.write(title_cell["source"].replace("#",
                                                         "").strip() + u"\n")
                    f.write(u"="*len(title_cell["source"])+u"\n\n")
                    f.write(notebook_template.substitute(name=fname_only,
                                                         body=html_out))
            hash_funcs.update_hash_dict(filehash, fname_only)
    except Exception, err:
        raise err

    finally:
        os.chdir(cur_dir)

    # probably not necessary
    del notebook_runner
