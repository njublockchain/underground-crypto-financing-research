import networkx

filename = "updated_graph_with_ahc" # CHANGEME
G: networkx.MultiDiGraph = networkx.read_gexf(f"key_node/{filename}.gexf")

networkx.write_latex(networkx.DiGraph(G), f"tex/{filename}.tex", as_document=True)
