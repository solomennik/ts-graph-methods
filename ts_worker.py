"""
Functions for working with TS in Gorban data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from itertools import combinations


############################################

def multiindex_resampler(df, freq, groups_lst, time_index_name):
    """
    Resamples Multiindex df with time series index,
    produces mean values

    df : pandas data frame
    freq : str
        resampling frequency - Timedelta
    groups_lst : str
        categorical variables defining groups to resample time
    time_index_name : str
        name of index variable with time series data

    """
    df = df.reset_index().set_index(time_index_name)
    df = df.dropna()
    return(df.groupby(groups_lst).resample(freq).mean().drop(groups_lst, axis=1))


def ts_multiplot(df, groups_var, columns_lst, figsize):
    """
    Plots multiple time series in data frames
    with DatetimeIndex

    df : pandas data frame with DatetimeIndex
    groups_var : str
        column to group time series
    columns_lst : list
        list of columns with values to plot on Y-axis
    figsize : set
        set of plot size for matplotlib plt.subplots() function
    """
    for price in columns_lst:
        f, ax = plt.subplots(figsize=figsize)

        for comp in set(df[groups_var]):
            ax.plot(df[df[groups_var]==comp][price], label=comp)

        plt.title(price)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.show()


def window_corrs(df, group, feature, start_date, end_date):
    """
    Prepares node list (nodes_lst) for node creation,
    feature and tuples lists (feature_lst, tuples_lst)
    for edges creation in a future graph 

    Parameters
    ----------
    df : pandas data frame with DatetimeIndex as 
        normal window
    group : str
        column with 'groups' - future nodes, 
        to count correlatin between
    feature : str
        variable with values to count correlations
    start_date : DatetimeIndex
        start date in time window
    end_date : DatetimeIndex
        end date in time window

    Returns
    ----------
    nodes_lst : list
        list with future node labels
    feature_lst ; lst
        list for future edges labels
    tuples_lst : list
        list with tuples as in feature_lst 
        values for proper edge construction
    n1 : list
        start node
    n2 : list
        end node
    weight : list
        abs values of correlation coeffs as weights

    """

    window = df.loc[(df.index >= start_date) & (df.index < end_date), 
        [group, feature]]

    normal_values = {}

    for i in list(set(window[group])):
        normal_values[i] = (window[window[group] == i][[feature]].to_numpy()).ravel()

    feature_lst = []
    n1 = []
    n2 = []
    weight = []

    for i in list(combinations(normal_values.keys(), r = 2)):

        a = list(i)
        feature_lst = feature_lst + [a[0] + '-' + a[1]]

        n1 = n1 + [a[0]]
        n2 = n2 + [a[1]]
        weight = weight + \
            [abs(np.corrcoef(normal_values[a[0]], normal_values[a[1]])[0][1])]
    
    weight = [0 if np.isnan(x) else x for x in weight]
    nodes_lst = list(normal_values.keys())
    tuples_lst = [tuple(i.split('-')) for i in feature_lst]

    return(nodes_lst, feature_lst, tuples_lst, n1, n2, weight)


def graph_from_dict(nodes_lst, n1, n2, weight):
    """
    Creates weighted graph from window_corr() function's
    return
    
    Parameters
    ----------
    nodes_lst : list
        list with future node labels
    feature_lst ; lst
        list for future edges labels
        not needed in this function, 
        should be passed with None
    tuples_lst : list
        list with tuples as in feature_lst 
        values for proper edge construction
        not needed in this function, 
        should be passed with None
    n1 : list
        start node
    n2 : list
        end node
    weight : list
        correlation coeffs as weights
    """

    G = nx.Graph() 

    G.add_nodes_from(nodes_lst)

    for n, nn, w in zip(n1, n2, weight):
        G.add_edge(n, nn, weight = w)
    
    return G



def graph_plotter(G, figsize=(10,10), title=''):

    """
    Creates circular graph plot with node size 
    (through weighted degree) and edge width depending
    on edge weights

    Parameters
    ----------
    G : networkx.classes.graph.Graph
    
    figsize : set
        set of plot size for matplotlib plt.subplots() function

    title : str
        plot title

    Returns
    ----------
    Plot of a graph

    """
    
    pos = nx.circular_layout(G)

    edge_labels = dict([((n1, n2), d['weight']) for 
        n1, n2, d in G.edges(data=True)])

    degree_dict = dict(G.degree(weight='weight'))

    plt.figure(figsize=figsize)
    plt.title(title)

    nx.draw(G, pos, with_labels = True, node_size=[v * 50 for v in degree_dict.values()])

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5)


    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2])

    plt.show()


def graph_chars(G, uin, normalized=False):
    """
    Computes various weighted graph characteristics
    Values can be normilized

    Parameters
    ----------
    G : networkx.classes.graph.Graph
    uin : str or int
        future primary key value
    normalized : bool
        If True the betweenness values are normalized using networkX
        parameters

    Returns
    ----------
    future_data : dictionary
        dictionary with ```uin```- primary key for future data
        processing
        varioous centrality parameters statistics (mean,
        sd, max, min, median) of a graph

    Further use
    -----------
    We can pass the dict to pandas DataFrame or to numpy matrix

    """

    try:
        clsns = nx.closeness_centrality(G, distance='weight', wf_improved=normalized).values()
    except:
        clsns = 0
    btwnns = nx.betweenness_centrality(G, weight='weight', normalized=normalized).values()
    edge_btwnns = nx.edge_betweenness_centrality(G, weight='weight', normalized=normalized).values()
    try:
        pgrnk = nx.pagerank(G, weight='weight', 
            max_iter=1000).values()
    except:
        pgrnk = 0
    try:
        eign = nx.eigenvector_centrality(G, weight='weight', 
            max_iter=1000).values()
    except:
        eign = 0
    auth = nx.hits(G)[1].values()
    strength = np.array(list(dict(G.degree(weight='weight')).values()))/len(G.nodes)

    try:
        flow_clsns = nx.current_flow_closeness_centrality(G, 
            weight='weight').values()
    except:
        flow_clsns = 0
    try:
        inforns = nx.information_centrality(G, weight='weight').values()
    except:
        inforns = 0
    acfbc = nx.approximate_current_flow_betweenness_centrality(G, weight='weight', normalized=normalized).values()
    ecfbc = nx.edge_current_flow_betweenness_centrality(G, weight='weight', normalized=normalized).values()
    cfbc = nx.current_flow_betweenness_centrality(G, weight='weight', normalized=normalized).values()
    loadc = nx.load_centrality(G, weight='weight', normalized=normalized).values()
    harm = nx.harmonic_centrality(G, distance='weight').values()

    future_data = {}

    future_data['id'] = uin

    for cent, name in zip([btwnns, clsns, edge_btwnns, pgrnk, eign, auth, strength, flow_clsns, inforns, \
            acfbc, ecfbc, cfbc, loadc, harm], \
        ['btwnns', 'clsns', 'edge_btwnns', 'pgrnk', 'eign', 'auth', 'strength', 'flow_clsns', 'inforns', \
            'acfbc', 'ecfbc', 'cfbc', 'loadc', 'harm']):
        try:
            future_data[name + '_mean'] = np.array(list(cent)).mean()
            future_data[name + '_sd'] = np.array(list(cent)).std()
            future_data[name + '_min'] = np.array(list(cent)).min()
            future_data[name + '_median'] = np.median(np.array(list(cent)))
            future_data[name + '_max'] = np.array(list(cent)).max()
        except:
            future_data['id'] = str(uin) + '_defect'

            future_data[name + '_mean'] = 999
            future_data[name + '_sd'] = 999
            future_data[name + '_min'] = 999
            future_data[name + '_median'] = 999
            future_data[name + '_max'] = 999

    return future_data