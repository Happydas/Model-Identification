import chart_studio.plotly as py
#import plotly.graph_objs
from plotly.graph_objs import *
py.sign_in('Happy_Das', 'v3MozO2Qhlwcpd95upsf')

samples = [100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000]
correct_prediction = [3, 8, 10, 10, 10, 10, 10, 10, 10, 10]
def visualize(sample, prediction):
    samp, avg_pred = [], []
    for a, b in zip(sample, prediction):
        samp.append(a)
        avg_pred.append(b)

    data1 = {
        "x": samp,
        "y": avg_pred
    }

    data = [data1]
    layout = {

        "title": "Correct Prediction against Samples for 10% noise",
        "xaxis": {"title": "Number of Samples"},
        "yaxis": {"title": "Correct Predictions"}
    }
    return data, layout
data, layout = visualize(samples, correct_prediction);
fig = Figure(data=data, layout=layout)
fig.update_layout(
    title={
        'text': " Number of correct predictions over samples for 10% noise",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
plot_url = py.plot(fig)