import os
from utils import (split_pdf_extract_data,create_chunked_embeddings,prepare_data_from_dataframe,create_index_definition,bulk_insert_data,generate_text_embedding,search_multiple_indexes,extract_text_from_search_results,create_clean_image_folder)
from opensearchpy import OpenSearch
import google.generativeai as genai
from vertexai.generative_models import GenerativeModel
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import streamlit as st  

generation_config = {
    "max_output_tokens": 8192,  # Reasonable for detailed responses, adjusted based on the typical length of financial reports.
    "temperature": 0.5,  # Lower value to ensure focused and precise output.
    "top_p": 0.95,  # Slightly lower to prioritize high-probability tokens for more accurate responses.
}
multimodal_model = GenerativeModel(
    "gemini-1.5-pro-001", generation_config=generation_config
)
genai.configure(api_key = "AIzaSyAHHqEMYWf3C-HmQI2xL2s-43jH0YTh0H8")
model = multimodal_model
folder_name = 'images'
folder_uri = os.path.join(os.getcwd(), folder_name)
if not os.path.exists(folder_uri):
    os.makedirs(folder_uri)
def create_rag(file_name, pdf_list):
    pdf_files = []
    folder_uri = 'images/'
    create_clean_image_folder(folder_uri)
    for pdf in pdf_list:
        # Save the PDF to a temporary file
        with open(os.path.join(folder_uri, pdf.name), "wb") as f:
            f.write(pdf.getbuffer())
        pdf_files.append(os.path.join(folder_uri, pdf.name))
    text_dataframe = split_pdf_extract_data(pdf_files,folder_uri)
    embedding_dataframe = create_chunked_embeddings(text_dataframe)
    data_for_opensearch = prepare_data_from_dataframe(embedding_dataframe,file_name)
    return data_for_opensearch

def create_indexes(client, data,index_name):
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    indices = {}
    indices.update(create_index_definition(index_name))
    for index_name, index_definition in indices.items():
        try:
            response = client.indices.create(index=index_name, body=index_definition)
            print(f"Created index: {index_name}, Response: {response}")
        except Exception as e:
            print(f"Failed to create index: {index_name}, Error: {str(e)}")
    bulk_insert_data(client, data, index_name)

def gemini_response(client,user_question,index_names):
    query = generate_text_embedding(user_question)
    query_body = {
        "query": {"knn": {"embedding": {"vector": query, "k": 10}}},
        "_source": False,
        "fields": ["id","text"],
        "size":10
        }
    search_results_list = [search_multiple_indexes(client, query_body, index) for index in index_names]
    text_results = extract_text_from_search_results(search_results_list)
    # st.write(text_results)
    template = [f"""SYSTEM:You are a financial report analyst, So you are provided with a financial reports of multiple companies.
                Use the context below to answer the question provided.
                <Question>
                {user_question}
                </Question>
                <Instructions>
                    -Analyze the Question: Identify the key variables such as specific financial metrics, time periods (e.g., Q2 FY24), and any other relevant -details mentioned in the question.
                    -Contextual Analysis: Review the provided context thoroughly to extract and analyze relevant data to answer the question accurately.
                    -Multi-Company Comparison: If the question involves comparing multiple companies, analyze the data for each company and provide a comparative answer.
                    -Focus and Precision: Answer strictly based on the question's requirements and the provided context.
                    -Provide a brief analyses of the answer you provided in the last.
                    --If an exact match for a financial metric with the same name isn't found in the context, please find a similar financial metric for that company and provide the answer to the question
                    -Read Whole Context.As it a long context and contains data for all quarters of each company. Don't skip data for any company in between.
                    -Sometime you will need to calculate things.(eg.headcount of Q1 - headcount of Q2 = People hired in Q2)
                </Instructions>
                <Suggestions>
                    -Key Figures Extraction: Focus on extracting and interpreting critical figures from the consolidated statements of comprehensive income.
                    -Visual Data Interpretation: Pay close attention to tables, graphs, and charts as they often summarize crucial data.
                    -Scope Limitation: Use only the provided context to derive your answer. If no data is available for a specific company or period, clearly   state, "No data uploaded for this company or period."
                    -Trend Analysis: Highlight significant trends or changes in performance compared to previous quarters.
                    -Concise Summarization: Provide a brief summary at the end, highlighting the key findings and insights from the analysis.
                    -The context will be large, That's why every chunk of context is named with company name and quarter so don't stop in between read the whole context
                    
                </Suggestions>
                4.Context starts from
         """]
    string = ""
    for result in text_results:
        template.append(string.join(result["text"]))
    p = " ".join(template)
    response = model.generate_content([user_question,p])
    answer = response.text
    return answer   




