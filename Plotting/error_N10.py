import chart_studio.plotly as py
#import plotly.graph_objs
from plotly.graph_objs import *
py.sign_in('Happy_Das', 'v3MozO2Qhlwcpd95upsf')

samples = [100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000]
xi1_error = [0.0183079, 0.00288071, 0.00280302, 0.00082023, 0.00019435, 0.00061994, 0.00011266, 0.00065684, 0.00033986, 0.00008628]
xi2_error = [0.1892245, 0.12299614, 0.03353519, 0.04834064, 0.02900468, 0.02863002, 0.04124718, 0.01890743, 0.01965981, 0.02786769]
def visualize(sample, averagexi1, averagexi2):
    samp, avg_xi1, avg_xi2 = [], [], []
    for a, b, c in zip(sample, averagexi1, averagexi2):
        samp.append(a)
        avg_xi1.append(b)
        avg_xi2.append(c)

    data1 = {
        "x": samp,
        "y": avg_xi1,

        "name": "$\\hat\\xi_{1e}$ ",
        "type": "scatter"
    }
    data2 = {
        "x": samp,
        "y": avg_xi2,

        "name": "$\\hat\\xi_{2e}$ ",
        "type": "scatter"
    }

    data = [data1, data2]
    layout = {

        "title": "Average value of $\\xi_1$ and $\\xi_2$ against number of Samples for 10% noise",
        "xaxis": {"title": "Number of Samples"},
        "yaxis": {"title": "Coefficients Error"}
    }
    return data, layout
data, layout = visualize(samples, xi1_error, xi2_error);
fig = Figure(data=data, layout=layout)
fig.update_layout(
    title={
        'text': "Average value of coefficients error against number of samples with 10% noise",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
plot_url = py.plot(fig)