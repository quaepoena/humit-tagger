# This is a simple web service wrapper for the tagger.
# When run, it runs a web interface at the local and if allowed at the external IP
# 
import sys
sys.path.append('..')
from flask import Flask, request
from tag import *
import io

# Load Models and config requred for tagging
load_models_and_config()

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    try:
        text =""
        tagged=""
        err=""
        lang ="au"
        lang_tag=""
        all_tagged=""
        all_err=""

        if request.method == 'POST':
            text= str(request.form['text']).strip()
            lang = str(request.form['lang'])
        elif request.method == 'GET':
            text = str(request.args['text']).strip()
            lang = str(request.args['lang'])

        # Just to check everything is supposed to be
        if lang=="":
            lang="au"
        if lang=="au":
            lang="au"
        elif lang=="bm":
            lang="bm"
        else:
            lang="nn"

        # Get tagging

        output_content = io.StringIO()
        lang_used = io.StringIO()

        sentences = tag(text, output_content, lang, False, lang_used , True)

        all_tagged = output_content.getvalue()
        lang_tag = lang_used.getvalue()
        output_content.close()
        lang_used.close()

        return {"lang":lang_tag,"sentences":sentences}, 200, {'Content-Type': 'application/json'}
    except:
        return {"error":"There is an error"}, 500,{'Content-Type': 'application/json'} 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000,debug=True)
