from dash import Dash, Input, Output, State, html, dcc
import dash
from dash_extensions import Keyboard
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pyperclip


def run_app(linspace, signal):
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True, pages_folder="", url_base_pathname="/3/")

    max_signal = np.max(signal)
    min_signal = np.min(signal)
    signal_span = np.abs(max_signal - min_signal)

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    colors = ["red", "blue", "green", "purple", "orange", "pink", "brown", "cyan", "magenta"]*3

    half_transparent = ["rgba(255, 0, 0, 0.5)", "rgba(0, 0, 255, 0.5)", "rgba(0, 255, 0, 0.5)", "rgba(128, 0, 128, 0.5)", "rgba(255, 165, 0, 0.5)", "rgba(255, 192, 203, 0.5)", "rgba(165, 42, 42, 0.5)", "rgba(0, 255, 255, 0.5)", "rgba(255, 0, 255, 0.5)"]*3

    class Cursor(dict):
        """Creates a cursor shape for the graph."""
        def __init__(self, letter):
            super().__init__()
            pos = linspace[int(len(linspace) * (alphabet.index(letter.lower()) + 5) / len(alphabet)) % len(linspace) + (5 if letter.isupper() else 0)]
            self["type"] = "line"
            self["x0"] = pos
            self["x1"] = pos
            self["y0"] = max_signal + 100*signal_span
            self["y1"] = min_signal - 100*signal_span
            self["visible"] = False
            self["line"] = dict(color=colors[alphabet.index(letter.lower())])
            self["label"] = dict(
                text=letter, 
                textposition="top center", 
                font=dict(size=20, color=colors[alphabet.index(letter.lower())])
            )
    
    class CursorMagnet(go.Scatter):
        """Creates a cursor magnet for the graph that snaps onto data points."""
        def __init__(self, letter):
            super().__init__()
            pos = linspace[int(len(linspace) * (alphabet.index(letter.lower()) + 5) / len(alphabet)) % len(linspace) + (5 if letter.isupper() else 0)]
            self.x = [pos]
            self.y = [signal[np.argmin(np.abs(linspace - pos))]]
            self.name = letter
            self.mode = "markers"
            self.marker = dict(size=10, color=colors[alphabet.index(letter.lower())])
            self.visible = False

    class CursorTableCol(dict):
        """Creates a column for the table that displays the cursor values."""
        def __init__(self, letter):
            super().__init__()
            self["headerName"] = f"Cursor {letter}"
            self["field"] = letter
            if letter.islower():
                self["colSpan"] = {"function": "params.data.property === 'Δx' || params.data.property === 'Δy' || params.data.property === 'Δy/Δx' ? 2 : 1"}
            self["valueFormatter"] = {"function": "d3.format(',.4f')(params.value)"}
            self["cellStyle"] = {'textAlign': 'center', 'background-color': half_transparent[alphabet.index(letter.lower())], 'color': 'black'}


    def layout(n_cursors=3):
        if not isinstance(n_cursors, int):
            if not n_cursors.isdigit():
                n_cursors = 3
            n_cursors = int(n_cursors)
        if n_cursors > 26:
            n_cursors = 26
        letters = alphabet[:n_cursors]
        return [
            dbc.Row([
                dcc.Graph(
                    id='graph', 
                    figure=go.Figure(
                        data=[
                            go.Scatter(x=linspace, y=signal, line=dict(color="black"), name="signal"),
                            *[CursorMagnet(letter) for letter in letters], # lower case cursor magnets
                            *[CursorMagnet(letter.upper()) for letter in letters], # upper case cursor magnets
                        ],
                    layout=go.Layout(
                        shapes = [
                            *[Cursor(letter) for letter in letters], # lower case cursors
                            *[Cursor(letter.upper()) for letter in letters], # upper case cursors
                        ],
                        yaxis=dict(
                            range=[min_signal - 0.1*signal_span, max_signal + 0.1*signal_span],
                            autorange=False,
                            mirror=True,
                            showline=True,
                            linewidth=2,
                            linecolor='black',
                        ),
                        xaxis=dict(
                            range=[linspace[0], linspace[-1]],
                            autorange=False,
                            mirror=True,
                            showline=True,
                            linewidth=2,
                            linecolor='black',
                        ),
                        showlegend=False, 
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        margin={k: 50 for k in ["l", "r", "b", "t"]},
                        )
                    ), config={"displayModeBar": True, "edits": {"shapePosition": True}, "doubleClick": "reset", "scrollZoom":True, 'modeBarButtonsToRemove': ['autoscale']}
                )
            ], style={"height": "60%", "width": "100%"}, className="g-0"),
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col([
                    dcc.Checklist(
                        id="check",
                        options=[{"label": html.Span(f"Cursors {letter}", style={"font-weight":600}), "value": letter} for letter in letters],
                        value=[],
                        inputStyle={"margin": "10px", "height":"20px", "width":"20px"},
                        style={"text-align": "center"},
                        labelStyle={"margin": "auto", "font-size": 20}
                    ),
                ], width=2, align="top"),
                dbc.Col(width=1),
                dbc.Col([
                    Keyboard( # Keyboard component to capture Ctrl+C
                        id="keyboard",
                        captureKeys=["c"],
                        eventProps=["ctrlKey", "key"],
                    ),
                    dag.AgGrid(
                        id="table",
                        columnDefs=[{"headerName": "Property", "field": "property", "pinned": "left"}],
                        rowData=[{"property": p } for p in ["y", "x", "Δy", "Δx", "Δy/Δx"]],
                        defaultColDef={"cellStyle": {'textAlign': 'center'}, "width": 100},
                        dashGridOptions = {"domLayout": "autoHeight", "rowSelection": "multiple"},
                        style = {"height": None, "width": "100%"},
                    ),
                    dbc.Alert( # Alert component to show breifly when the table content is copied to clipboard
                        "Table content copied to clipboard!",
                        id="alert",
                        is_open=False,
                        duration=2000,
                        color="white",
                    ),
                ], width=6)
            ], className="g-0",style={"marginRight": "30px", "marginLeft": "30px", "marginBottom": "5px", "height": "40%"}),
        ]

    dash.register_page("", path="/3", path_template='/<n_cursors>', layout=layout, title="Multi Cursors") # Register the layout function as a page to use <n_cursors> in the URL

    app.layout = dbc.Container(dash.page_container, fluid=True, class_name="g-0")

    @app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Output("table", "columnDefs"),
        Input("check", "value"),
        State("graph", "figure"),
        State("table", "columnDefs"),
        prevent_initial_call=True
    )
    def show_cursors(values, fig, col_defs):
        """
        Update the visibility of the cursors and cursor magnets based on the selected values.
        Also add the columns for the selected cursors to the table.
        """
        col_defs_to_keep = [col_def for col_def in col_defs if col_def["field"].lower() in values + ["property"]]
        for value in values:
            if not any(col["field"].lower() == value for col in col_defs_to_keep):
                col_defs_to_keep.extend([
                    CursorTableCol(value), CursorTableCol(value.upper())
                ])
        for shape in fig["layout"].get("shapes", []):
            shape["visible"] = shape["label"]["text"].lower() in values
        for scatter in fig["data"]:
            scatter["visible"] = scatter["name"].lower() in values + ["signal"]
        return fig, col_defs_to_keep

    @app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("graph", "relayoutData"),
        Input("graph", "figure"),
        prevent_initial_call=True
    )
    def update_cursor_magnets(_, fig):
        """
        Update the position of the cursor magnets based on the position of the cursors and the data points.
        """
        for i, shape in enumerate(fig["layout"].get("shapes", [])):
            index = np.argmin(np.abs(linspace - shape["x1"]))
            fig["data"][i + 1]["x"] = [linspace[index]]
            fig["data"][i + 1]["y"] = [signal[index]]
        return fig
    
    @app.callback(
        Output("table", "rowData"),
        Input("graph", "figure"),
        State("table", "rowData"),
        prevent_initial_call=True
    )
    def update_table(fig, row_data):
        """
        Update the values in the table based on the position of the cursors.
        """
        lower_cursors = [magnet for magnet in fig["data"][1:] if magnet["name"].islower()]
        upper_cursors = [magnet for magnet in fig["data"][1:] if magnet["name"].isupper()]

        for lower, upper in zip(lower_cursors, upper_cursors):
            lower_name = lower["name"]
            upper_name = upper["name"]
            x_lower, y_lower = lower["x"][0], lower["y"][0]
            x_upper, y_upper = upper["x"][0], upper["y"][0]
            y_delta = y_upper - y_lower
            x_delta = x_upper - x_lower
            slope = y_delta / x_delta if x_delta else np.inf
            row_data[0][lower_name] = y_lower
            row_data[0][upper_name] = y_upper
            row_data[1][lower_name] = x_lower
            row_data[1][upper_name] = x_upper
            row_data[2][lower_name] = y_delta
            row_data[3][lower_name] = x_delta
            row_data[4][lower_name] = slope
        return row_data

    @app.callback(
        Output("alert", "is_open"),
        Input("keyboard", "n_keydowns"),
        State("keyboard", "keydown"),
        State("table", "rowData"),
        State("table", "columnDefs"),
        prevent_initial_call=True
    )
    def copy_to_clipboard(n, event, row_data, col_defs):
        """
        Copy the whole table content to the clipboard when Ctrl+C is pressed.
        """
        if event["key"] == "c" and event["ctrlKey"]:
            headers = [col_def["field"] for col_def in col_defs]
            data = "\r\n".join(["\t".join([str(row.get(header, "")) for header in headers]) for row in row_data])
            pyperclip.copy("\t".join(headers) + "\r\n" + data)
            return True
        return False

    app.run(debug=True)
