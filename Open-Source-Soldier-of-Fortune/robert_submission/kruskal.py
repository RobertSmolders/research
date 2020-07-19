import pandas as pd 
import numpy as np
import networkx as nx

def correlation(prices, reverse=False):
    """
    Calculates the correlation matrix from asset prices

    :param prices: (pd.DataFrame) Asset price data
    :param reverse: (bool)  When True, the data order of the data is
                            reversed to cope with different data 
                            indexing. (False by default)
    :return: (pd.DataFrame) Correlation matrix
    """
    
    #Calculate returns
    #Take reverse order of index into account
    if reverse == False:
        returns = np.log(prices.divide(prices.shift(1))).iloc[1:,:]
    elif reverse == True:
        returns = np.log(prices.divide(prices.shift(-1))).iloc[:-1,:]

    #Calculate correlation matrix
    correlation = pd.DataFrame(
                    np.nan, index=returns.columns, 
                    columns=returns.columns
                    )

    #Iterating over tickers
    for i in range(0, returns.shape[1]):
        
        #Iterating over tickers, skipping half the matrix due to symmetry
        for j in range(i, returns.shape[1]):

            #Calculate numerator
            num = (returns.iloc[:,i].multiply(returns.iloc[:,j]).mean() 
                    - returns.iloc[:,i].mean() * returns.iloc[:,j].mean())

            #Calculate denominator and correlation coefficient
            if i == j:
                denom = (np.power(returns.iloc[:,i], 2).mean() 
                        - np.power(returns.iloc[:,i].mean(), 2))
                correlation.iloc[i, i] = num / abs(denom)

            elif i != j:
                denom_i = (np.power(returns.iloc[:,i], 2).mean() 
                            - np.power(returns.iloc[:,i].mean(), 2))
                denom_j = (np.power(returns.iloc[:,j], 2).mean() 
                            - np.power(returns.iloc[:,j].mean(), 2))
                corr = num / np.power(denom_i*denom_j, 0.5)

                correlation.iloc[i,j], correlation.iloc[j,i] = corr, corr

    return correlation

def distance(correlation):
    """
    Calculates the distance matrix from a correlation matrix

    :param prices: (pd.DataFrame) Correlation matrix
    :return: (pd.DataFrame) Distance matrix
    """

    #Create pandas DataFrame of ones
    ones = pd.DataFrame(
                np.repeat(
                    1, correlation.shape[1]**2
                    ).reshape(
                        correlation.shape[1], 
                        correlation.shape[1]
                        ), 
                index=correlation.columns, columns=correlation.columns
                )

    #Calculate distance
    distance = np.power(ones - correlation, 0.5) * 2**0.5

    return distance

def MST(distance):
    """
    Calculates the MST according to Kruskal's algorithm

    :param distance: (pd.DataFrame) Distance matrix
    :return: (networkx.Graph) Minimum Spanning Tree
    """

    #List all distances and their corresponding tickers
    node_1, node_2, index = [], [], []

    #Iterate over tickers
    for i in range(0, distance.shape[1]):

        #Iterating over tickers, skipping half the matrix due to symmetry 
        for j in range(i, distance.shape[1]):

            #Skip distance of oneself, since this distance is zero
            if i != j:
                node_1.append(distance.columns[i])
                node_2.append(distance.columns[j])
                index.append(distance.iat[i,j])

    #Put lists in a DataFrame and sort by ascending distance
    dis_sort = pd.DataFrame(
                {'0': node_1, '1': node_2}, index=index
                ).sort_index(
                    ascending=True
                    )

    #Initialize graph and nodes
    G = nx.Graph()
    G.add_nodes_from(distance.columns)

    #Create edges based on ascending distance according to Kruskal's algorithm
    for i in range(0, len(dis_sort.index)):
        
        #For each distance, check if a path between tickers already exists, 
        #if not then create an edge between the tickers
        if (
            dis_sort.iat[i, 0] not in 
            list(nx.algorithms.descendants(G, dis_sort.iat[i, 1]))
            ):
            #Add edge between nodes
            G.add_edge(dis_sort.iat[i, 0], dis_sort.iat[i, 1])

    return G