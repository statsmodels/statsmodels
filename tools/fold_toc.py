#!/usr/bin/env python
import sys
import re

# Read doc to string
filename = sys.argv[1]
try:
    static_path = sys.argv[2]
except:
    static_path = '_static'
doc = open(filename).read()

# Add mktree to head
pre = '<head>'
post = '''<head>
    <script type="text/javascript" src="static_path/mktree.js"></script>
    <link rel="stylesheet" href="static_path/mktree.css" type="text/css">
'''
post = re.sub('static_path', static_path, post)
doc = re.sub(pre, post, doc)

# TOC class
pre = '''<div class="toctree-wrapper compound">
<ul>'''
post = '''<div class="toctree-wrapper compound">
<a onclick="expandTree('toctree#')" href="javascript:void(0);">Expand all. </a>
<a onclick="collapseTree('toctree#')" href="javascript:void(0);">Collapse all.</a>
<ul class="mktree" id="toctree#">'''
toc_n = doc.count('toctree-wrapper')
for i in range(toc_n):
    post_n = re.sub('#', str(i), post)
    doc = re.sub(pre, post_n, doc, count=1)

## TOC entries
pre = '<li class="toctree-l1">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# TOC entries 2nd level
pre = '<li class="toctree-l2">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# TOC entries 3rd level
pre = '<li class="toctree-l3">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# TOC entries 4th level
pre = '<li class="toctree-l4">'
post = '<li class="liClosed"> '
doc =  re.sub(pre, post, doc)

# Write to file
f = open(filename, 'w')
f.write(doc)
f.close()
