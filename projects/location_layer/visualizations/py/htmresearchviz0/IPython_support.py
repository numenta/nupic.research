import json
import numbers
import os
import uuid

from pkg_resources import resource_string

from IPython.display import HTML, display


def header_content():
  path = os.path.join('package_data', 'htmresearchviz0-bundle.js')
  htmresearchviz0_js = resource_string('htmresearchviz0', path).decode('utf-8')


  return f"""
<style>

div.htmresearchviz-output {{
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;

  padding-bottom: 2px;
}}

div.htmresearchviz-output svg {{
  max-width: initial;
}}

</style>
<script>
  var nupic_undef_define = ("function"==typeof define),
      nupic_prev_define = undefined;
  if (nupic_undef_define) {{
    nupic_prev_define = define;
    define = undefined;
  }}
  {htmresearchviz0_js}
  if (nupic_undef_define) {{
    define = nupic_prev_define;
  }}
</script>"""



def init_notebook_mode():
    display(HTML(header_content() + """
    <script>
      if (window.nupicQueue) {{
        window.nupicQueue.forEach(f => f());
        window.nupicQueue = null;
      }}
    </script>
    """))


def getAddChartHTML(elementId, renderCode):
    return f"""
    <div class="htmresearchviz-output" id="{elementId}"></div>
    <script>
    (function() {{
      function render() {{
        {renderCode}
      }}

      if (window.htmresearchviz0) {{
        render();
      }} else {{
        if (!window.nupicQueue) {{
          window.nupicQueue = [];
        }}
        window.nupicQueue.push(render);
      }}
    }})();
    </script>
    """


def printSingleLayer2DExperiment(csvText):
    elementId = str(uuid.uuid1())
    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.printSingleLayer2DExperiment(document.getElementById('%s'), '%s');" % (
        elementId, csvText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printLocationModuleInference(logText):
    elementId = str(uuid.uuid1())
    
    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.locationModuleInference.printRecording(document.getElementById('%s'), '%s');" % (
        elementId, logText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printLocationModulesRecording(logText):
    elementId = str(uuid.uuid1())

    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.locationModules.printRecording(document.getElementById('%s'), '%s');" % (
        elementId, logText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printMultiColumnInferenceRecording(logText):
    elementId = str(uuid.uuid1())

    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.multiColumnInference.printRecording(document.getElementById('%s'), '%s');" % (
        elementId, logText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printPathIntegrationUnionNarrowingRecording(logText):
    elementId = str(uuid.uuid1())

    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.pathIntegrationUnionNarrowing.printRecording(document.getElementById('%s'), '%s');" % (
        elementId, logText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printSpikeRatesSnapshot(jsonText):
    elementId = str(uuid.uuid1())
    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.insertSpikeRatesSnapshot(document.getElementById('%s'), '%s');" % (
        elementId, jsonText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printSpikeRatesTimeline(jsonText):
    elementId = str(uuid.uuid1())
    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.insertSpikeRatesTimeline(document.getElementById('%s'), '%s');" % (
        elementId, jsonText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printInputWeights(jsonText):
    elementId = str(uuid.uuid1())
    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.insertInputWeights(document.getElementById('%s'), '%s');" % (
        elementId, jsonText.replace('\r', '\\r').replace('\n', '\\n'))
    )))


def printOutputWeights(jsonText):
    elementId = str(uuid.uuid1())
    display(HTML(getAddChartHTML(
      elementId,
      "htmresearchviz0.insertOutputWeights(document.getElementById('%s'), '%s');" % (
        elementId, jsonText.replace('\r', '\\r').replace('\n', '\\n'))
    )))
