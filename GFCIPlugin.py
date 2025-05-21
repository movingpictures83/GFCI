#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# #### Load the example dataset

# In[2]:

class GFCIPlugin:
 def input(self, inputfile):
  data_dir = inputfile
  self.df = pd.read_table(data_dir, sep="\t")

 def run(self):
  pass

 def output(self, outputfile):
  # #### Start Java VM

  # In[3]:


  from pycausal.pycausal import pycausal as pc
  pc = pc()
  pc.start_vm(java_max_heap_size = '1000M')


  # #### Create the Prior Knowledge Object

  # In[4]:


  from pycausal import prior as p
  forbid = [['history_noise','class'],['history_fluctuating','class']]
  tempForbid = p.ForbiddenWithin(
    ['class','history_fluctuating','history_noise'])
  temporal = [tempForbid]
  prior = p.knowledge(forbiddirect = forbid, addtemporal = temporal)
  prior


  # #### Load causal algorithms from the py-causal library and Run Bootstrapping GFCI Discrete

  # In[5]:


  from pycausal import search as s
  tetrad = s.tetradrunner()
  tetrad.listIndTests()


  # In[6]:


  tetrad.listScores()


  # In[7]:


  tetrad.getAlgorithmParameters(algoId = 'gfci', testId = 'disc-bic-test', scoreId = 'bdeu-score')


  # In[8]:


  tetrad.run(algoId = 'gfci', dfs = self.df, testId = 'disc-bic-test', scoreId = 'bdeu-score', 
           priorKnowledge = prior, dataType = 'discrete',
           maxDegree = 3, maxPathLength = -1, 
           completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = True,
           numberResampling = 5, resamplingEnsemble = 1, addOriginalDataset = True)


  # #### Bootstrapping GFCI Discrete's Result's Nodes

  # In[9]:


  tetrad.getNodes()


  # #### Bootstrapping GFCI Discrete's Result's Edges

  # In[10]:


  tetrad.getEdges()


  # #### Plot The Result's Graph

  # In[11]:


  import pydot
  #from IPython.display import SVG
  outf = open(outputfile+".txt", 'w')
  dot_str = pc.tetradGraphToDot(tetrad.getTetradGraph())
  outf.write(dot_str)
  graphs = pydot.graph_from_dot_data(dot_str)
  graphs[0].write_png(outputfile)
  #svg_str = graphs[0].create_svg()
  #SVG(svg_str)


  # #### Stop Java VM

  # In[12]:


  pc.stop_vm()


  # In[ ]:




