
import os
import fitz  # PyMuPDF
import time
#from PIL import Image  # Assuming you use PIL or Pillow for image processing
import pandas as pd
from opensearchpy import OpenSearch
import json
from opensearchpy import helpers
import uuid
from vertexai.generative_models import GenerativeModel, Image
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import (
    GenerativeModel,
    Image
)
from langchain_community.document_loaders import DataFrameLoader
import vertexai
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"
PROJECT_ID = "crypto-handbook-425405-f1"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


vertexai.init(project=PROJECT_ID, location=LOCATION)
generation_config = {
    "max_output_tokens": 8192,  # Reasonable for detailed responses, adjusted based on the typical length of financial reports.
    "temperature": 0.5,  # Lower value to ensure focused and precise output.
    "top_p": 0.95,  # Slightly lower to prioritize high-probability tokens for more accurate responses.
}
multimodal_model = GenerativeModel(
    "gemini-1.5-pro-001", generation_config=generation_config
)
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
# Pass The folder path for storing the images
def create_clean_image_folder(Image_Path):
    # Create the directory if it doesn't exist
    if not os.path.exists(Image_Path):
        os.makedirs(Image_Path)
folder_uri = 'images/'
create_clean_image_folder(folder_uri)

def split_pdf_extract_data(pdfList, folder_uri):
    # To get better resolution
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

    # Create a folder for each PDF if it doesn't exist
    for indiv_Pdf in pdfList:
        pdf_filename = os.path.basename(indiv_Pdf)
        print(pdf_filename)
        pdf_filename_without_extension = os.path.splitext(pdf_filename)[0]
        pdf_folder_path = os.path.join(folder_uri, pdf_filename_without_extension)
        if not os.path.exists(pdf_folder_path):
            os.makedirs(pdf_folder_path)

        doc = fitz.open(indiv_Pdf)  # open document
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            outpath = os.path.join(pdf_folder_path, f"{pdf_filename_without_extension}_{page.number}.png")
            pix.save(outpath)  # store image as a PNG

    # Define the path where images are located
    image_names = os.listdir(pdf_folder_path)
    print(image_names)
    Max_images = len(image_names)

    # Create empty lists to store image information
    page_source = []
    page_content = []
    page_id = []

    p_id = 0  # Initialize image ID counter
    rest_count = 0  # Initialize counter for error handling

    while p_id < Max_images:
        try:
            # Construct the full path to the current image
            image_path = os.path.join(pdf_folder_path, image_names[p_id])
            # Skip directories
            if os.path.isdir(image_path):
                p_id += 1
                continue

            # Load the image (assuming you have a method or library for image loading)
            print(f'opening image {image_path}')
            image = Image.load_from_file(image_path) #.tobytes()
            print(f'Image Opened -{image_path}')

#             # Generate prompts for text and table extraction
            prompt_text =(
                "Extract all text content in the image"
            )
            prompt_table =(
                "Detect table in this image. Extract content maintaining the structure"
                )

            # Example of how to use your multimodal model for content extraction
            # Replace with your actual method or model for content generation
            contents_text = [image, prompt_text]
            response_text = multimodal_model.generate_content(contents_text)
            text_content = response_text.text

            contents_table = [image, prompt_table]
            response_table = multimodal_model.generate_content(contents_table)
            table_content = response_table.text

            print(f"Processed image no: {p_id}")
            page_source.append(image_path)
            page_content.append(
                text_content + "\n" + table_content
            )  # Add image_content if enabled
            page_id.append(p_id)
            p_id += 1

        except Exception as err:
            # Handle errors during processing
            print(err)
            print("Taking Some Rest")
            time.sleep(25)  # Pause execution for 12 seconds due to default Quota for Vertex
            rest_count += 1
            if rest_count == 8:  # Limit consecutive error handling
                rest_count = 0
                print(f"Cannot process image no: {image_path}")
                p_id += 1  # Move to the next image

    df = pd.DataFrame({
        "page_id": page_id,
        "page_source": page_source,
        "page_content": page_content
    })
    del page_id, page_source, page_content
    shutil.rmtree(pdf_folder_path) 
    return df

def generate_text_embedding(text) -> list:
    """Text embedding with a Large Language Model."""
    embeddings = text_embedding_model.get_embeddings([text])
    vector = embeddings[0].values
    return vector

def create_chunked_embeddings(df):
    # Load documents from the DataFrame
    loader = DataFrameLoader(df, page_content_column="page_content")
    documents = loader.load()

    print(f"# of documents loaded (pre-chunking) = {len(documents)}")

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increasing chunk size to accommodate larger content
    chunk_overlap=200,  # Slightly larger overlap to ensure context preservation
    # separators=["\n\n", "\n", " ", ".", ",", ";", ":", "|", "-", "?", "!"],  # Adjusted to include common separators in reports
)

    # Split the documents
    doc_splits = text_splitter.split_documents(documents)
    for idx, split in enumerate(doc_splits):
        split.metadata["chunk"] = idx

    print(f"# of documents after splitting = {len(doc_splits)}")

    texts = [doc.page_content for doc in doc_splits]
    id_list = [str(uuid.uuid4()) for _ in range(len(doc_splits))]
    page_source_list = [doc.metadata["page_source"] for doc in doc_splits]
    
    # Function to generate embeddings in parallel
    def process_embedding(doc, idx):
        time.sleep(1)
        return generate_text_embedding(doc.page_content)

    # Using ThreadPoolExecutor for parallel processing
    text_embeddings_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_doc = {executor.submit(process_embedding, doc, idx): idx for idx, doc in enumerate(doc_splits)}
        for future in as_completed(future_to_doc):
            idx = future_to_doc[future]
            try:
                embedding = future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
                embedding = None
            text_embeddings_list.append(embedding)
    
    # Create the embeddings DataFrame
    embedding_df = pd.DataFrame(
        {
            "id": id_list,
            "embedding": text_embeddings_list,
            "page_source": page_source_list,
            "text": texts,
        }
    )
    
    return embedding_df
def prepare_data_from_dataframe(df, index_name):
    embeds_data = []
    
    for _, row in df.iterrows():
        data = {
            "_index": index_name,
            "id": row["id"],
            "page_source": row.get("page_source", ""),
            "text": row["text"],
            "embedding": row["embedding"]
        }
        embeds_data.append(data)
    return embeds_data

def create_index_definition(index_name):
    index_definition = {
        index_name: {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    }
                }
            }
        }
    }
    return index_definition
def bulk_insert_data(client, data, index_name):
    try:
        # Perform bulk insertion
        success, failed = helpers.bulk(
            client,
            data,
            index=index_name,
            raise_on_error=True,
            refresh=True
        )
        print(f"Bulk insert into {index_name} completed with {success} successes and {len(failed)} failures.")
    except helpers.BulkIndexError as e:
        print(f"Bulk insert into {index_name} failed with BulkIndexError: {e}")
    except Exception as e:
        print(f"Failed with general Exception: {e}")
def search_multiple_indexes(client, query_body, index_name):
    try:
        results = client.search(
            index=index_name,
            body=query_body
        )
        return results
    except Exception as e:
        print(f"Error occurred during search: {str(e)}")
        return None

def extract_text_from_search_results(search_results_list):
    text_results = []
    for search_results in search_results_list:
        if "hits" in search_results and "hits" in search_results["hits"]:
            for hit in search_results["hits"]["hits"]:
                # Extract the text from fields
                fields = hit.get("fields", {})
                index_name = hit.get("_index", "unknown_index")  # Get the index name
                
                if "id" in fields:
                    # Concatenate the index name to the text
                    combined_text = f"Following data belongs to the: {index_name}\n\n{fields['text'][0]}"
                    text_results.append({
                        "text": combined_text
                    })
    
    return text_results

