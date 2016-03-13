import time
from bokeh.client import push_session
from bokeh.plotting import (curdoc, figure)

session = push_session(curdoc())
session.show()
# create the figure
fig = figure(title='testing',
             x_axis_label='iterations',
             y_axis_label='value',
             logo=None,
             toolbar_location='right')
time.sleep(.1)
# create a new line
l = fig.line([], [], legend='test_line', name='test_line',
            line_color='#1f77b4')
# stream some data
time.sleep(3)
for i in range(100):
    l.data_source.stream({'x':[i],'y':[i]})
    time.sleep(.05)
