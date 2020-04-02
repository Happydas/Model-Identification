import chart_studio.plotly as py
#import plotly.graph_objs
from plotly.graph_objs import *
py.sign_in('Happy_Das', 'v3MozO2Qhlwcpd95upsf')

samples = [100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000]
xi1 = [0.0816921, 0.09711929, 0.09719698, 0.09917977, 0.10019435, 0.09938006, 0.09988734, 0.09934316, 0.10033986, 0.10008628]
xi2 = [0.8107755, 0.87700386, 0.96646481, 0.95165936, 0.97099532, 0.97136998, 0.95875282, 0.98109257, 0.98034019, 0.97213231]
def visualize(sample, averagexi1, averagexi2):
    samp, avg_xi1, avg_xi2 = [], [], []
    for a, b, c in zip(sample, averagexi1, averagexi2):
        samp.append(a)
        avg_xi1.append(b)
        avg_xi2.append(c)

    data1 = {
        "x": samp,
        "y": avg_xi1,

        "name": "$\\hat\\xi_{1}$ ",
        "type": "scatter"
    }
    data2 = {
        "x": samp,
        "y": avg_xi2,

        "name": "$\\hat\\xi_{2}$ ",
        "type": "scatter"
    }

    data = [data1, data2]
    layout = {

        "title": "Average value of $\\xi_1$ and $\\xi_2$ against number of Samples for 10% noise",
        "xaxis": {"title": "Number of Samples"},
        "yaxis": {"title": "Coefficients"}
    }
    return data, layout
data, layout = visualize(samples, xi1, xi2);
fig = Figure(data=data, layout=layout)
fig.update_layout(
    title={
        'text': "Average value of coefficients against number of samples with 10% noise",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
plot_url = py.plot(fig)