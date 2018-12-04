import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

with open('/tmp/Output.json', 'r') as file:
    graph_object = json.loads(file.read())

for index, island in enumerate(graph_object['islands']):
    plt.title('Result for {} cities, {} islands and {} generations ; island {}'.format(len(graph_object['cities']),
                                                                                       len(graph_object['islands']),
                                                                                       graph_object['nGeneration'],
                                                                                       index))
    fig = plt.figure()
    xdata = []
    ydata = []
    for island_index in island:
        xdata.append(graph_object['cities'][island_index][0])
        ydata.append(graph_object['cities'][island_index][1])
    plt.plot(xdata, ydata)
    fig.savefig('graph'+str(index)+'.png')
    plt.close(fig)
