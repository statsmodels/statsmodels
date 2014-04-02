#!/usr/bin/env python

"""
Python script for building documentation.

To build the docs you must have all optional dependencies for statsmodels
installed. See the installation instructions for a list of these.

Note: currently latex builds do not work because of table formats that are not
supported in the latex generation.

Usage
-----
python make.py clean
python make.py html
"""

import os
import shutil
import sys
import subprocess
import base64

os.environ['PYTHONPATH'] = '..'

SPHINX_BUILD = 'sphinxbuild'


def upload():
    sf_account = 'jseabold,statsmodels@web.sourceforge.net'
    retcode = subprocess.call(['rsync', '-avPrzh', '--inplace', '-e ssh',
                               'build/html/',
                               sf_account + ':htdocs/vbench/'],
                               stderr=sys.stderr, stdout=sys.stdout)

    if retcode != 0:
        msg = "Could not upload vbench result"
        raise Exception(msg)


def clean():
    if os.path.exists('build'):
        shutil.rmtree('build')

    if os.path.exists('source/generated'):
        shutil.rmtree('source/generated')


def html():
    check_build()
    if os.system('sphinx-build -P -b html -d build/doctrees '
                 'source build/html'):
        raise SystemExit("Building HTML failed.")


def check_build():
    build_dirs = [
        'build', 'build/doctrees', 'build/html',
        'build/plots', 'build/_static',
        'build/_templates']
    for d in build_dirs:
        try:
            os.mkdir(d)
        except OSError:
            pass


def all():
    clean()
    html()


def auto_update():
    msg = ''
    try:
        clean()
        html()
        upload()
        sendmail()
    except (Exception, SystemExit) as inst:
        msg += str(inst) + '\n'
        sendmail(msg)


def sendmail(err_msg=None):
    if err_msg is None:
        msgstr = 'vbench uploaded successfully'
        subject = "VB: update successful"
    else:
        msgstr = err_msg
        subject = "VB: update failed"

    server_str, port, login, pwd = _get_credentials()

    import smtplib
    from email.MIMEText import MIMEText
    msg = MIMEText(msgstr)
    msg['Subject'] = subject
    msg['From'] = login
    msg['To'] = login

    to_email = [login]
            #('josef.pktd' + 'AT' + 'gmail' + '.com').replace('AT', '@')]

    server = smtplib.SMTP(server_str, port)
    server.ehlo()
    server.starttls()
    server.ehlo()

    server.login(login, pwd)
    try:
        server.sendmail(login, to_email, msg.as_string())
    finally:
        server.close()


def _get_dir(subdir=None):
    import getpass
    USERNAME = getpass.getuser()
    if sys.platform == 'darwin':
        HOME = '/Users/%s' % USERNAME
    else:
        HOME = '/home/%s' % USERNAME

    if subdir is None:
        subdir = '/code/scripts'
    conf_dir = '%s%s' % (HOME, subdir)
    return conf_dir


def _get_credentials():
    # my security holes
    with open('/home/skipper/statsmodels/gmail.txt') as f:
        pwd = f.readline().strip()
    pwd = base64.b64decode(pwd)
    email_name = 'statsmodels.dev' + 'AT' + 'gmail' + '.com'
    email_name = email_name.replace('AT', '@')

    return 'smtp.gmail.com', 587, email_name, pwd


funcd = {
    'html': html,
    'clean': clean,
    'upload': upload,
    'auto_update': auto_update,
    'all': all,
}

small_docs = False

# current_dir = os.getcwd()
# os.chdir(os.path.dirname(os.path.join(current_dir, __file__)))

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        func = funcd.get(arg)
        if func is None:
            raise SystemExit('Do not know how to handle %s; valid args are %s' % (
                arg, funcd.keys()))
        func()
else:
    small_docs = False
    all()
# os.chdir(current_dir)
