import pandas as pd
import numpy as np
import os
import re
import json
import obonet
import inflect
import networkx as nx
from openai import OpenAI
from pydantic import BaseModel
# import google.generativeai as genai
from dotenv import load_dotenv
from itertools import chain
load_dotenv()
from configparser import ConfigParser

config = ConfigParser()
configFilePath = 'data/app.cfg'
with open(configFilePath) as f:
    config.read_file(f)
API_KEY = os.environ.get('API_KEY')
LUNG_REFERENCE = config['documents']['lung_reference']

## Define output format
class cellTypeFormat(BaseModel):
       cellType: str
class cellTypeListFormat(BaseModel):
       cellType: list[str]
## Read cell clusters file
## file path set to ./Data/all.csv
def read_cluster(path):
      dataframe = pd.read_csv(path)
      print(dataframe['marker'])
      return dataframe

# def repeat_dataframe_with_index(df, n):
#     # Repeat the DataFrame n times
#     repeated_df = pd.concat([df.copy() for _ in range(n)], ignore_index=True)
#     # Add a repeat index column
#     repeat_indices = [i for i in range(n) for _ in range(len(df))]
#     repeated_df.insert(0, 'sround', repeat_indices)
#     return repeated_df

#gene_list = repeat_dataframe_with_index(gene_list, 5)
#print(gene_list)
## Split genelist
## split genelist to multiple chunks so that it won't exceed model input size
# Generate the user message for each chunk
def generate_user_message(gene_list:  pd.DataFrame, max_chunk_size=100) -> list:
    # Split the gene_list into chunks
    chunks = [group for _, group in gene_list.groupby(gene_list.index // max_chunk_size)]
    messages = []
    cluster_names= []
    for chunk in chunks:
        # Construct the user message for each chunk
        user_message = (
            chunk['marker']
        )
        cluster_name = chunk['cluster_name']
        messages.append(user_message)
        cluster_names.append(cluster_name)
    return cluster_names, messages


client = OpenAI(api_key=API_KEY)
### send prompt to GPT
marker_importance=pd.read_csv(LUNG_REFERENCE)
#marker_importance=pd.read_csv('{}importance_neuroblastoma.csv'.format(WORK_DIR))
#poor_marker=pd.read_csv(f'{WORK_DIR}poor_performance_genes.csv')
#poor_marker_lst=poor_marker['marker'].tolist()
#marker_importance=marker_importance[~marker_importance['marker'].isin(poor_marker_lst)]
#print(reference)
def extract_markers_in_message(message):
  markers=[]
  for row in message:
    #marker_str=row.split(':')[1]
    marker_str=row
    marker_lst=marker_str.split(',')
    marker_lst=[marker.replace('"','').replace("'","").strip() for marker in marker_lst]
    markers.extend(marker_lst[0:5])
  return set(markers)

def get_relevant_row(message, df):
  markers=extract_markers_in_message(message)
  #print(markers)
  #print(markers)
  hints = []
  for marker in markers:
    #print(marker)
    #Filter the DataFrame for the given marker
    filtered_df = df[df['marker'] == marker]
    #print(filtered_df)
    if not filtered_df.empty:
        output = ', '.join(
            f"{row['cell_type']} (p={row['p_val_adj']:.1e})"
            for _, row in filtered_df.iterrows()
        )
        hints.append(f"{marker} is significant in {output}")
    else:
        hints.append("")
  return "\n".join(hints)

def generate_hint(message, df, top_rank=5):
  markers=extract_markers_in_message(message)
  #print(markers)
  #print(markers)
  hints = []
  for marker in markers:
    #print(marker)
    #Filter the DataFrame for the given marker
    filtered_df = df[df['marker'] == marker]
    #print(filtered_df)
    if not filtered_df.empty:
        output = '\n'.join(
            f"marker: {marker}; cell type: {row['cell_type']}; importance: {row['importance']}\n"
            for _, row in filtered_df.sort_values(by='importance', ascending=False)[0:top_rank].iterrows()
        )
        hints.append(output)
    else:
        hints.append(f"marker: {marker}; No relevant information")
  return "\n".join(hints)

def annotateCell_GPT_with_hint(messages: list, model: str)->list:
    global client
    global marker_importance
    responses = []
    for message in messages:
        introduction="Use the below information about markers of cell types to answer the following questions, smaller p-value indicates more importance of the marker in the cell type"
        hint = generate_hint(message,marker_importance)
        question = """Identify cell types using the following tissue name and markers separately for each row.
Only provide the cell type name.
Do not show numbers before the name.
Some can be a mixture of multiple cell types."""
        content = f"{introduction}\n{hint}\n{question}"
        trial=0
        while trial<5:
            # Use JSON response format for other models
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": content},
                    {"role": "user", "content": str(message)}
                ],
                response_format=cellTypeListFormat,
            )
            response_cells = completion.choices[0].message.parsed.cellType
            response_cells = [ re.sub(r'^[^a-zA-Z]+', '', response_cell) for response_cell in response_cells ]
            # try again if response length not equal to message
            if len(response_cells)==len(message):
                break
            trial=trial+1
        responses.extend(response_cells)
        print(message)
        print(response_cells)
    return responses

def RAG1(input_dict):
    global client
    global marker_importance
    #print(input_dict)
    result = {}
    gene_list=pd.DataFrame(list(input_dict.items()), columns=["cluster_name", "marker"])
    #print(gene_list)
    cluster_names, messages = generate_user_message(gene_list,10)
    #print(cluster_names)
    #print(messages)
    cluster_names = list(chain(*cluster_names))
    responses = annotateCell_GPT_with_hint(messages, 'gpt-4o-2024-08-06')
    result = dict(zip(cluster_names, responses))
    print(result)
    return result
