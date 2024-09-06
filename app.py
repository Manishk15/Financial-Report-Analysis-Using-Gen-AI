import streamlit as st
from streamlit_functions import create_rag, create_indexes, gemini_response
from opensearchpy import OpenSearch

# Configuration for OpenSearch
VM_IP = '10.128.0.14'  # VM's public IP address
PORT = '9200'
CLUSTER_URL = f'https://{VM_IP}:{PORT}'

def get_client(cluster_url=CLUSTER_URL,
               username='admin',  # Your OpenSearch username
               password='admin',  # Your OpenSearch password
               use_ssl=True,      # True if using HTTPS
               verify_certs=False # Set to True to verify SSL certificates in production
               ):
    client = OpenSearch(
        hosts=[cluster_url],
        http_auth=(username, password),
        use_ssl=use_ssl,
        verify_certs=verify_certs
    )
    return client

client = get_client()
import json
import os
import streamlit as st

INDEX_NAMES_FILE = "/tmp/index_names.json"

# Function to load index names from the JSON file
def load_index_names():
    if os.path.exists(INDEX_NAMES_FILE):
        with open(INDEX_NAMES_FILE, 'r') as file:
            return set(json.load(file))
    return set()

# Function to save index names to the JSON file
def save_index_names(index_names):
    with open(INDEX_NAMES_FILE, 'w') as file:
        json.dump(list(index_names), file)
# List of system indices to preserve
system_indices = [
    '.opendistro_security',
    '.opensearch-observability',
    '.plugins-ml-config',
    '.kibana',  # Example for Kibana-related indices
    # Add any other known system indices here
]
def main():
    st.header("Financial Report Analyst")

    # Initialize the index names in session state if not already done
    if 'index_names' not in st.session_state:
        st.session_state.index_names = load_index_names()  # Load from file if available

    with st.sidebar:
        st.title("Menu:")
        pdf_list = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",accept_multiple_files = True)

        if st.button("Submit & Process"):
            if pdf_list:
                with st.spinner("Processing..."):
                    for pdf in pdf_list:
                        name = pdf.name
                        file_name = name[:-4].lower()

                        if file_name not in st.session_state.index_names:
                            # Process and index the PDF only if it hasn't been processed before
                            data = create_rag(file_name, [pdf])
                            create_indexes(client, data, file_name)
                            st.session_state.index_names.add(file_name)  # Add to session state and JSON file
                            st.write(f"Processed '{file_name}'.")
                        else:
                            st.write(f"'{file_name}' is already uploaded.")
                    # Save the updated index names to the JSON file
                    save_index_names(st.session_state.index_names)

                    st.success("Processing complete.")
                    st.write(f"Currently uploaded files: {', '.join(st.session_state.index_names)}")
            else:
                st.warning("Please upload at least one PDF file.")
        indices = client.indices.get('*')
        if st.button("Uploaded Financial Reports"):
            for index in indices:
                if index not in system_indices and not index.startswith('security-auditlog'):
                    st.write(index)
        index_to_delete = st.selectbox("Select a file to delete", options=[i for i in indices if i not in system_indices and not i.startswith('security-auditlog')])
        if st.button("Delete Selected File"):
            if index_to_delete:
                client.indices.delete(index=index_to_delete)
                st.session_state.index_names.discard(index_to_delete)  # Remove from session state
                save_index_names(st.session_state.index_names)
                st.success(f"Index '{index_to_delete}' deleted successfully.")
            else:
                st.warning("Please select a file to delete.")
        if st.button("Delete All Files"):
            indices = client.indices.get('*')
            for index in indices:
                if index not in system_indices and not index.startswith('security-auditlog'):
                    client.indices.delete(index=index)
            st.success("All files deleted successfully.")
            st.session_state.index_names.clear()  # Clear session state
            save_index_names(st.session_state.index_names)       
    # User input for querying
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        if st.session_state.index_names:
            response = gemini_response(client, user_question, list(st.session_state.index_names))
            st.write(response)
        else:
            st.warning("No files Uploaded!!.Please upload and process PDF files first.")
    
     

if __name__ == "__main__":
    main()
