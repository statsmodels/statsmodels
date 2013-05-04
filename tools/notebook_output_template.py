from string import Template

notebook_template = Template("""
.. raw:: html

$body

   <script src="https://c328740.ssl.cf1.rackcdn.com/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"type="text/javascript"></script>
   <script type="text/javascript">
   init_mathjax = function() {
       if (window.MathJax) {
           // MathJax loaded
           MathJax.Hub.Config({
               tex2jax: {
               // I'm not sure about the \( and \[ below. It messes with the
               // prompt, and I think it's an issue with the template. -SS
                   inlineMath: [ ['$$','$$'], ["\\\\(","\\\\)"] ],
                   displayMath: [ ['$$$$','$$$$'], ["\\\\[","\\\\]"] ]
               },
               displayAlign: 'left', // Change this to 'center' to center equations.
               "HTML-CSS": {
                   styles: {'.MathJax_Display': {"margin": 0}}
               }
           });
           MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
       }
   }
   init_mathjax();

   // since we have to load this in a ..raw:: directive we will add the css
   // after the fact
   function loadcssfile(filename){
       var fileref=document.createElement("link")
       fileref.setAttribute("rel", "stylesheet")
       fileref.setAttribute("type", "text/css")
       fileref.setAttribute("href", filename)

       document.getElementsByTagName("head")[0].appendChild(fileref)
   }
   // loadcssfile({{pathto("_static/nbviewer.pygments.css", 1) }})
   // loadcssfile({{pathto("_static/nbviewer.min.css", 1) }})
   loadcssfile("../../../_static/nbviewer.pygments.css")
   loadcssfile("../../../_static/nbviewer.min.css")
   </script>""")
