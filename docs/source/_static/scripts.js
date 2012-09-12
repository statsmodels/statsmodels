function cleanUpText(codebox){
    /// Not currently used
    /// Strips a whole IPython session of input and output prompts
    //escape quotation marks
    codebox = codebox.replace(/"/g, "\'");

    // newlines
    codebox = codebox.replace(/[\r\n|\r|\n]$/g, ""); // remove at end
    codebox = codebox.replace(/[\r\n|\r|\n]+/g, "\\n");
    // prompts
    codebox = codebox.replace(/In \[\d+\]: /g, "");
    codebox = codebox.replace(/Out \[\d+\]: /g, "");

return codebox;
}

function htmlescape(text){
    return (text.replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;"))
}

function scrapeText(codebox){
    /// Returns input lines cleaned of prompt1 and prompt2
    var lines = codebox.split('\n');
    var newlines = new Array();
    $.each(lines, function() {
        if (this.match(/^In \[\d+]: /)){
            newlines.push(this.replace(/^(\s)*In \[\d+]: /,""));
        }
        else if (this.match(/^(\s)*\.+:/)){
            newlines.push(this.replace(/^(\s)*\.+: /,""));
        }

    }
            );
    return newlines.join('\\n');
}

$(document).ready(            
        function() {
    // grab all code boxes
    var ipythoncode = $(".highlight-ipython");
    $.each(ipythoncode, function() {
        var codebox = scrapeText($(this).text());
        // give them a facebox pop-up with plain text code   
        $(this).append('<span style="text-align:left; display:block; margin-top:-10px; margin-left:10px; font-size:75%"><a href="javascript: jQuery.facebox(\'<textarea cols=80 rows=10 readonly style=margin:5px onmouseover=javascript:this.select();>'+htmlescape(htmlescape(codebox))+'</textarea>\');">View Code</a></span>');
        $(this,"textarea").select();
    });
});
